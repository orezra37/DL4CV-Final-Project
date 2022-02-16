import argparse
import yaml
import numpy as np
from pathlib import Path

# torch imports
import torch
from torch import autograd
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from openfold.utils.tensor_utils import tensor_tree_map

# openfold imports
from openfold.np.residue_constants import ID_TO_HHBLITS_AA

# my imports
from StudentDataset import StudentDataset
from our_models.EvoStudent import EvoStudent
from our_models.OG_former import OG1

MAX_CAPACITY_LENGTH = 400
EPS = 1

dataset_id_to_aatype = {
    0: "A",
    1: "R",
    2: "N",
    3: "D",
    4: "C",
    5: "Q",
    6: "E",
    7: "G",
    8: "H",
    9: "I",
    10: "L",
    11: "K",
    12: "M",
    13: "F",
    14: "P",
    15: "S",
    16: "T",
    17: "W",
    18: "Y",
    19: "V",
    20: "-",
}


def train(args):
    """Train loop for OG-former based on EVO-former."""

    # torch.set_default_dtype(torch.float16)
    Path(f'./checkpoints/EvoStudent/{args.name}').mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter('logs/EvoStudent/' + args.name)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_dataset = StudentDataset(args.train_data)
    val_dataset = StudentDataset(args.val_data)
    # test_dataset = OGDataset("/path/to/test/data")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset)
    # test_loader = DataLoader(test_dataset)

    criterion = MSELoss()
    model = EvoStudent(num_blocks=config["num_blocks"], train=True)
    # model.half()
    model.cuda()
    # for i in range(len(model.evoformer.blocks)):
    #     # gpu_id = int(i // (config["num_blocks"] / 2))
    #     model.evoformer.blocks[i].to(f'cuda:{gpu_id}')
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), eps=1e-4)

    for e in range(config['epochs']):
        print(e)
        model.train()
        avg_train_loss = 0
        avg_train_s_loss = 0
        avg_train_z_loss = 0
        for i_batch, batch in enumerate(train_loader):
            x, y = batch
            target_s, target_z = y

            fetch_cur_batch = lambda t: t[..., 0]
            feats = tensor_tree_map(fetch_cur_batch, x)
            if feats['target_feat'].size(1) > MAX_CAPACITY_LENGTH:
                continue
            # Enable grad iff we're training and it's the final recycling layer

            for k, v in feats.items():
                if k != "extra_msa":
                    feats[k] = v #.half()
                feats[k].cuda()

            pred_s, pred_z = model(feats)

            # s_loss = criterion(pred_s, target_s.half())
            # z_loss = criterion(pred_z, target_z.half())

            # normalize so loss won't explode
            pred_s_norm = pred_s#(pred_s - target_s.mean()) / (target_s.std() + EPS)
            pred_z_norm = pred_z #(pred_z - target_z.mean()) / (target_z.std() + EPS)
            target_s_norm = target_s #(target_s.half() - target_s.mean()) / (target_s.std() + EPS)
            target_z_norm = target_z #(target_z.half() - target_z.mean()) / (target_z.std() + EPS)

            s_loss = 1e-5 * criterion(pred_s_norm, target_s_norm) #.half()
            z_loss = 1e-2 * criterion(pred_z_norm, target_z_norm) #.half()
            print('s loss:\t', s_loss.item(), '\tz loss:\t', z_loss.item(), '\tlen: \t', feats['target_feat'].size(1))
            loss = s_loss + z_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
            avg_train_s_loss += s_loss.item()
            avg_train_z_loss += z_loss.item()
            # logging
            if i_batch % config['train_logging_frequency'] == 0 and i_batch != 0:
                writer.add_scalar('train/avg_loss', avg_train_loss / config["train_logging_frequency"], i_batch)
                writer.add_scalar('train/avg_s_loss', avg_train_s_loss / config["train_logging_frequency"], i_batch)
                writer.add_scalar('train/avg_z_loss', avg_train_z_loss / config["train_logging_frequency"], i_batch)
                avg_train_loss = 0
                avg_train_s_loss = 0
                avg_train_z_loss = 0

            # if i_batch % config['val_logging_frequency'] == 0:
            #     print('evaluating...')
            #     model.eval()
            #     avg_val_loss = 0
            #     avg_val_acc = 0
            #     # val_log = open(f'./checkpoints/EvoEncoder/{args.name}/val_epoch_{e}_batch_{i_batch}.txt', 'w')
            #     for _, val_batch in enumerate(val_loader):
            #         x, y = val_batch
            #         target_s, target_z = y
            #         # print(f'len:\t{x[0].size()}')
            #         # seq_len = x[0].size(1)
            #         # if seq_len > MAX_CAPACITY_LENGTH:
            #         #     continue
            #         pred_s, pred_z = model(x)
            #         loss = criterion(pred_s, target_s) + criterion(pred_z, target_z)
            #         loss = criterion(probs, y[0])
            #         seq_len = probs.size(0)
            #         correct = (pred == y).sum().item()
            #         accuracy = correct / seq_len
            #         avg_val_loss += loss.item()
            #         avg_val_acc += accuracy
            #         # if accuracy < 1:
            #         #     val_log.write(f'*******************')
            #         # val_log.write(f'protein name: {name}\n')
            #         # val_log.write(f'accuracy: {accuracy * 100}%\n')
            #         # val_log.write(f'GT:\t{GT_str}\n')
            #         # val_log.write(f'pred:\t{pred_str}\n\n')
            #     # val_log.close()
            #
            #     if best_val_accuracy < avg_val_acc:
            #         best_val_accuracy = avg_val_acc
            #         torch.save(model, f'./checkpoints/EvoEncoder/{args.name}/best_model.pth')
            #         with open(f'./checkpoints/EvoEncoder/{args.name}/log.txt', 'w') as f:
            #             f.write(f'best model saved on epoch {e} and batch {i_batch}\naccuracy: '
            #                     f'{best_val_accuracy * 100}% for {len(val_loader)} proteins')
            #
            #     writer.add_scalar('val/loss', avg_val_loss / len(val_loader), i_batch)
            #     writer.add_scalar('val/accuracy', avg_val_acc * 100 / len(val_loader), i_batch)
            #     model.train()
            #
            # if i_batch % config['checkpoint_frequency'] == 0 and i_batch != 0:
            #     torch.save(model, f'./checkpoints/EvoEncoder/{args.name}/epoch_{e}_batch_{i_batch}.pth')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--debug', action='store_true')
    p.add_argument('--config', required=True, type=str)
    p.add_argument('--train_data', required=True, type=str, help='path to dataset train folder')
    p.add_argument('--val_data', required=True, type=str, help='path to dataset validation folder')
    p.add_argument('--name', type=str, default='untitled', help='experiment name (for checkpoints and logs)')
    args = p.parse_args()
    if args.debug:
        import pydevd_pycharm

        pydevd_pycharm.settrace('10.96.3.64', port=123, stdoutToServer=True, stderrToServer=True)
    train(args)
