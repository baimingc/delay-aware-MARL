import glob
import json
import matplotlib.pyplot as plt

file_list = glob.glob('./models/*/*/*/logs/*.json')
file_list = [name for name in file_list if '5' in name]
# file_list = [name for name in file_list if 'GAN-I' in name]
# file_list = [name for name in file_list if 'R_0.1__' in name]
# mode = 'full'  # full, all
# mode = 'test'  # full, all
# legend_lable = []

colormap = plt.cm.gist_ncar
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(file_list))])
print(file_list)
for name in file_list:
    with open(name, 'r') as f:
        returns = json.load(f)
        print(returns['models\\simple_tag\\log\\run5\\logs/agent0/mean_episode_rewards/reward'])


# plt.legend(legend_lable)
# plt.show()
