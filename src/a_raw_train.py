import torch
from torch import nn
import sys
from src import a_raw_models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

from torch.utils.tensorboard import SummaryWriter



####################################################################
#
# Construct the model 
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(a_raw_models, hyp_params.model+'Model')(hyp_params)
    #model = torch.load('/mnt/hard2/bella/erc/pretrained_models/at_30hist_1e-6.pt')
    if hyp_params.use_cuda:
        model = model.cuda()
    
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    #model.load_state_dict(torch.load('/mnt/hard2/bella/erc/pretrained_models/trial_30hist_1e-6'))
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, samples in enumerate(train_loader):
            text = samples['source']['text_inputids']
            text_mask = samples['source']['text_mask']
            audio = samples['source']['audio_raw']
            audio_mask = samples['source']['audio_mask']
            # text: [B x T_text]
            # audio: [B x T_audio x 74]
            # vision: [B x T_vision x 80 x 80]
            #print(samples['target'].shape)
            eval_attr = samples['target'].squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, eval_attr = text.cuda(), audio.cuda(), eval_attr.cuda()
                    text_mask, audio_mask = text_mask.cuda(), audio_mask.cuda()
                    if hyp_params.dataset == 'IEMOCAP':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            #batch_chunk = hyp_params.batch_chunk
            
                
            combined_loss = 0
            
            #print('audio vision:',audio.shape, vision.shape)
            preds= model(text, audio, text_mask, audio_mask)
            #if hyp_params.dataset == 'IEMOCAP':
                #preds = preds.view(-1, 2)
                #eval_attr = eval_attr.view(-1)
            #print(preds.shape, eval_attr.shape, preds, eval_attr)
            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss 
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                
                proc_loss, proc_size = 0, 0
                start_time = time.time()
   
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
             for i_batch, samples in enumerate(loader):
                text = samples['source']['text_inputids']
                text_mask = samples['source']['text_mask']
                audio = samples['source']['audio_raw']
                audio_mask = samples['source']['audio_mask']
                # text: [B x T_text]
                # audio: [B x T_audio x 74]
                # vision: [B x T_vision x 80 x 80]
                #print(samples['target'].shape)
                eval_attr = samples['target'].squeeze(-1)   # if num of labels is 1
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, eval_attr = text.cuda(), audio.cuda(), eval_attr.cuda()
                        text_mask, audio_mask = text_mask.cuda(), audio_mask.cuda()
                        if hyp_params.dataset == 'IEMOCAP':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                preds = model(text, audio, text_mask, audio_mask)
                '''
                if hyp_params.dataset == 'IEMOCAP':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                '''
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                preds_label=torch.argmax(preds, -1)
                results.append(preds_label)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        #results = torch.cat(results)
        #truths = torch.cat(truths)
        return avg_loss, results, truths
    
    writer = SummaryWriter()
    best_valid = 1e8
    #checkpoint=1
    #if checkpoint:
        #val_loss, _, _ = evaluate(model, criterion, test=False)
        #best_valid = val_loss
    
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train_loss = train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        writer.add_scalar("Loss/train", train_loss, epoch) 
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)   
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print("Saved model")
            save_model(hyp_params, model, name='hubem_30hist_1e-6')
            best_valid = val_loss
    
    #save_model(hyp_params, model, name='t_30hist_1e-6')
    model = load_model(hyp_params, model, name='hubem_30hist_1e-6')
    _, results, truths = evaluate(model, criterion, test=True)
    '''
    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'IEMOCAP':
        print('metrics')
        eval_iemocap(results, truths)
    '''
    results = torch.cat(results)
    truths = torch.cat(truths)
    #print(results.shape, truths.shape)
    test_preds = results.cpu().detach().numpy()
    test_truth = truths.cpu().detach().numpy()
    f1_weighted = f1_score(test_truth, test_preds, average='weighted')
    f1_perclass = f1_score(test_truth, test_preds, average=None)
    acc = accuracy_score(test_truth, test_preds)
    print("  - F1 weighted: ", f1_weighted)
    print("  - F1 per class: ", f1_perclass)
    print("  - Accuracy: ", acc)
    sys.stdout.flush()
    input('[Press Any Key to start another run]')
