import sys
sys.path.append('../')

from models.siren_metabase import Siren
from LSUN_meta_training_with_partition import MAML_multi_partition, LSUN_mask
from LSUN_meta_training import MAML
from util import *
from torch.utils.data import DataLoader
import argparse


def MAML_train(meta_model, data, model_name='test'):
    steps_til_summary = 10   # 100
    optim = torch.optim.Adam(lr=1e-4, params=meta_model.parameters())
    for step, sample in enumerate(data):
        sample = dict_to_gpu(sample)
        model_output = meta_model(sample)
        loss = ((model_output['model_out'] - sample['query']['y']) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

            torch.save(meta_model.state_dict(), './trained_models/'+model_name+'.pt')

        optim.zero_grad()
        loss.backward()
        optim.step()


def evaluate(meta_model, data):
    psnr_list = []
    for step, sample in enumerate(data):
        # We find that the batch size of evaluation must be the same as training.
        # Therefore, we repeat the sample to meet the batch size of training.
        sample = sample_repeat(sample,times=4)
        sample = dict_to_gpu(sample)
        model_output = meta_model(sample)

        totol_psnr = calc_psnr(model_output['model_out'], sample['query']['y'])
        psnr_list.append(totol_psnr.item())
        print(step, totol_psnr.item())

        # the MAML inference costs memory so that we use a backward to free it.
        # the parameters of the model would not be updated.
        totol_psnr.backward()

    print('psnr mean: ', np.mean(psnr_list))
    print('psnr std: ', np.std(psnr_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML_fit')

    # config
    parser.add_argument('--mode', type=str, required=False, default='train',
                        help='choose from [train, val]')
    parser.add_argument('--MAML_partition', action='store_true',
                        help='Whether using partition in the MAML')
    parser.add_argument('--data_root', type=str, required=False, default='./',
                        help='path of data root')
    parser.add_argument('--mask', type=str, required=False, default='None',
                        help='mask, options:[grid, hfs]')

    parser.add_argument('--trained_model_path', type=str, required=False, default='./test.pt',
                        help='the path to the model that needs to be evaluated.')

    args = parser.parse_args()

    img_siren = Siren(in_features=2, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True).cuda()

    if args.MAML_partition:
        assert (args.mask == 'grid' or args.mask == 'hfs'), 'must choose a mask method for partition.'

    if args.MAML_partition:
        model_name = 'partitionMAML' + '_' + args.mask
        meta_model = MAML_multi_partition(num_meta_steps=3, hypo_module=img_siren, loss=l2_loss, init_lr=1e-5,
                                          lr_type='per_parameter_per_step_single_head').cuda()
    else:
        model_name = 'MAML' + '-' + args.mask
        meta_model = MAML(num_meta_steps=3, hypo_module=img_siren, loss=l2_loss, init_lr=1e-5,
                          lr_type='per_parameter_per_step').cuda()

    n_subdomain = 4
    if args.mode == 'train':
        dataset = LSUN_mask(root=args.data_root, classes='church_outdoor_train',
                            n_subdomain=n_subdomain, mask_method=args.mask)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

        MAML_train(meta_model, data=dataloader, model_name=model_name)

    elif args.mode == 'val':
        dataset = LSUN_mask(root=args.data_root, classes='church_outdoor_val',
                            n_subdomain=n_subdomain, mask_method=args.mask)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        model_state_dict = torch.load(args.trained_model_path)
        meta_model.load_state_dict(model_state_dict)

        evaluate(meta_model, data=dataloader)






