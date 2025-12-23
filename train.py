from recbole.quick_start import run_recbole

if __name__ == '__main__':
    # dataset='splatoon3' と指定すると dataset/splatoon3/ フォルダを探しに行きます
    run_recbole(
        model='FM', 
        dataset='splatoon3', 
        config_file_list=['config.yaml']
    )
