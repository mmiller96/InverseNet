import sys
#sys.path.append("/Desktop/MasterArbeit/SindyAutoencoders/src")
from grid_trainer import Grid_Trainer

if __name__ == '__main__':
    grid_trainer = Grid_Trainer('analysis_alpha2')
    grid_trainer.process()
