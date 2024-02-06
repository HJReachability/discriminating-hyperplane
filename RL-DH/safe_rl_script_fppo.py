import subprocess

seeds = [0, 3, 4, 5, 6]
exps = ['pret_ppo']
envs = ['HalfCheetah']
steps_map = {
    'fppo': 5e6,
    'ppo': 1e6,
    'ppo_lag': 5e5,
    'pret_ppo': 1e6

}

initial = "python fppo.py --env HalfCheetah --seed 0 --steps 1e7"
commands = []
for env in envs:
    for seed in seeds:
        for exp in exps:
            steps = steps_map[exp]
            if exp == 'fppo':
                command = f'python fppo.py --env {env} --seed {seed} --exp_name {exp} --steps {steps}'
            elif exp == 'pret_ppo':
                pret_dir = f'data/fppo/fppo-{env}_s{seed}/pyt_save/model.pt'
                command = f'python ppo.py --env {env} --seed {seed} --exp_name {exp} --steps {steps} --pret_dir {pret_dir}'
            else:
                command = f'python ppo.py --env {env} --seed {seed} --exp_name {exp} --steps {steps}'
            print(command)
            commands.append(command)



if input("Exec above commands? (y/n): ") == 'y':
    for command in commands:
        try:
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)