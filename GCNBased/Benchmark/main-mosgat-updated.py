""" Example for MOSGAT classification, now looping over all three
    encoder variants (GAT, GraphSAGE, GIN) so you get results for all
    three instead of just the GAT default.
"""
from train_test_modified import train_test


if __name__ == "__main__":
    data_folder_list = ['KIPAN','ROSMAP','LGG','BRCA'] # here to run all four
    encoder_list = ['GAT', 'SAGE', 'GIN']

    testonly = False
    view_list = [1, 2, 3]
    num_epoch_pretrain = 500
    num_epoch = 1000
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-4

    for data_folder in data_folder_list:
        if data_folder == 'ROSMAP':
            num_class = 2
        if data_folder == 'LGG':
            num_class = 2
        if data_folder == 'KIPAN':
            num_class = 3
        if data_folder == 'BRCA':
            num_class = 5

        for encoder in encoder_list:
            print(f"\n{'='*60}")
            print(f"Dataset: {data_folder}  |  Encoder: {encoder}")
            print(f"{'='*60}")

            # early_stop_patience is optional; set to None to disable.
            # GIN benefits most from it since it overfits fastest (see
            # the GIN class docstring in models.py for why).
            early_stop = 200 if encoder == 'GIN' else None

            train_test(data_folder, view_list, num_class,
                       lr_e_pretrain, lr_e, testonly,
                       num_epoch_pretrain, num_epoch,
                       encoder=encoder,
                       early_stop_patience=early_stop,
                       method_label=f"MOSGAT_{encoder}")
