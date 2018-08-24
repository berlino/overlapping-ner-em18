import torch.nn
import torch.optim as optim

def adjust_learning_rate(optimizer):
    cur_lr = optimizer.param_groups[0]['lr']
    # adj_lr = cur_lr / 2
    adj_lr = cur_lr * 0.1
    print("Adjust lr to ", adj_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr

def create_opt(parameters, config):
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters,lr=config.lr, rho=config.rho, eps=config.eps, weight_decay=config.l2)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr, weight_decay=config.l2)
    return optimizer


def clip_model_grad(model, clip_norm):
    torch.nn.utils.clip_grad_norm(model.parameters(), clip_norm, norm_type=2)


# misc_config is dic that is generated according to dataset
def load_dynamic_config(misc_dict, config):
    config.voc_size = misc_dict["voc_size"]
    config.char_size = misc_dict["char_size"]
    config.pos_size = misc_dict["pos_size"]
    config.label_size = misc_dict["label_size"]

    print(config) # print training setting