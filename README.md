# Branching DQN
> Branching DQN implementation with pytorch based on https://github.com/seolhokim/BipedalWalker-BranchingDQN. 
> It is also sufficiently capable of showing (almost) optimal movements after 1000 episodes in BipedalWalker-v3 environment.
> For better performance in BipedalWalker-v3, I use some tricks mentioned in https://zhuanlan.zhihu.com/p/409553262.
> However it seems fine in other environments without these tricks. :)

## Dependencies
python==3.9.10  
gym==0.18.3  
torch==1.13.1  
*Other versions may also work well. It's just a reference.*  

## Structure
**/data:** contains results of training or testing, including graphs and videos  
**/model:** contains pre-trained models

  
## Train
use:

```bash
python train.py
```

- **--round | -r :** training rounds (default: 2000)
- **--tensorboard | -t :** use tensorboard  
- **--lr_rate | -l :** learning rate (default: 0.0001)
- **--batch_size | -b :** batch size (default: 64)
- **--gamma | -g :** discounting factor gamma (default: 0.99)
- **--action_scale | -a :** discrete action scale among the continuous action space (default: 50)
- **--env | -e :** environment to train in (default: BipedalWalker-v3)
- **--per | -p :** use per  
- **--load | -l :** to specify the model to load in ./model/ (e.g. 2000 for [env]_2000.pth)  
- **--save_interval | -s :** interval round to save model(default: 100)
- **--print_interval | -d :** interval round to print evaluation(default: 50)


## Test
use:
```bash
python enjoy.py
```

- **--not_render | -n :** not to render
- **--round | -r :** for evaluation rounds (default: 10)
- **--action_scale | -a :** for discrete action scale (default: 50)
- **--load | -l :** to specify the model to load in ./model/ (e.g. 2000 for [env]_2000.pth)
- **--env | -e :** for which environment to test in (default: BipedalWalker-v3)

P.S. *It is highly recommended to use same **action_scale** and **env** in training and testing. Otherwise, the performance in testing could be rather unpredictable.*

## Performance
> **Scores in Training:**  
![Score in 2000 episodes](data/score.png)  
> **Trained Model:**  
![Visual performance](data/render.gif)