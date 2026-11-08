"""Run the multidevice experiment from a JSON configuration."""
import argparse,json
from moire_project import multidevice as experiment

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/multidevice_train.json')
    args=parser.parse_args()
    with open(args.config,encoding='utf-8') as f:settings=json.load(f)
    for key,value in settings.items():
        if not key.isupper() or not hasattr(experiment,key):raise ValueError('Unknown configuration key: '+key)
        setattr(experiment,key,value)
    experiment.run()

if __name__=='__main__':main()
