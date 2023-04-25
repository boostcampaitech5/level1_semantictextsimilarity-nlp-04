import argparse

def get_args():
    # 기본 설정값만 남기고, hyperparameter는 wandb에게 맡김.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-base', type=str)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--loss', default='L1', help="['L1','MSE','Huber','plcc'] 중 택1", type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--patience', default=5, type=int, help="Early stopping 조건. Correlation coefficient 개선이 [patience]번 없으면 종료")

    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    
    parser.add_argument('--ckpt_path', default='./checkpoints')

    # wandb에 전달할 config
    parser.add_argument('--run_count', default=10, type=int, help="wandb sweep 실험 몇번 진행할건지?")

    args = parser.parse_args(args=[])
    return args
