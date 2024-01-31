import numpy  as np
import os
import torch 
class total_model_dict():
    def __init__(self, model_config=None, training_config=None, data_config=None):
        super(total_model_dict, self).__init__()
        self.dictionary = dict()

        # Create dictionary object with model class data
        if model_config is not None:
            self.dictionary['Model Configuration'] = model_config

        # Create dictionary object with training settings
        if model_config is not None:
            self.dictionary['Training Configuration'] = training_config

        # Create dictionary object with data settings
        if model_config is not None:
            self.dictionary['Data Configuration'] = data_config
    
    def update_loss(self, loss_list):
        # Iterate over Layer 1 Keys
        for key_name in loss_list.keys():
            
            # Check if we are appending a list (e.g. Epoch Times) or a dictionary (e.g. Loss Statistics)
            # Create List keys accordingly
            if key_name not in self.dictionary.keys():
                if isinstance(loss_list[key_name], dict): 
                    self.dictionary[key_name] = dict()
                    for key_name2 in loss_list[key_name]:
                        if key_name2 not in self.dictionary[key_name].keys():
                            self.dictionary[key_name][key_name2] = list()

                elif isinstance(loss_list[key_name], (int, float, complex)):
                    self.dictionary[key_name] = list()
            
            # Append items as appropriate
            if isinstance(loss_list[key_name], dict): 
                for key_name2 in loss_list[key_name]:
                    self.dictionary[key_name][key_name2].append(loss_list[key_name][key_name2])
            
            elif isinstance(loss_list[key_name], (int, float, complex)):
                self.dictionary[key_name].append(loss_list[key_name])

    def update_statistics(self, x, name):
        self.update_loss({name: {'Mean' : np.mean(x), 'Std Dev' : np.std(x), 'Max' : np.max(x), 'Min' : np.max(x)}})

    def __iter__(self):
        return self.dictionary
    
def save_checkpoint(path, name, model=None, loss_dict=None, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if model != None:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        if optimizer is not None:
            optim_dict = optimizer.state_dict()
        else:
            optim_dict = 0.0

        torch.save({
            'model': model_state_dict,
            'optim': optim_dict
        }, ckpt_dir + name + '.pt')
        print('Checkpoint is saved at %s' % ckpt_dir + name + '.pt')

    if loss_dict != None:
        np.save(ckpt_dir + name + '_results', loss_dict)
        print("Training Dictionary Saved in Same Location")