import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class RLEvaluation:
    def __init__(self, episode_ticks = 1, mean_subset = 100, logfile='log.txt', resume_train=False):
        self.episodes, self.loss_values, self.score_values, self.mean_scores = [], [], [], []
        self.episode_ticks = episode_ticks

        self.mean_subset = mean_subset
        

        #init plots
        sns.set(style="whitegrid")
        plt.clf()
        plt.ion()
        self.loss_plot = plt.subplot(211)
        self.score_plot = plt.subplot(212)
        self.loss_plot.set_ylabel("loss values")
        self.loss_plot.set_xlabel("episodes")
        self.score_plot.set_ylabel("score values")
        self.score_plot.set_xlabel("episodes")
        self.logfile=logfile
        if resume_train is False:
            self.write_log_header()
        else:
            self.load_training_data()

    def plot_train_loss(self):
        self.loss_plot.semilogy(self.episodes, self.loss_values)
        plt.draw()
        plt.pause(0.001)

    def plot_score(self):
        self.score_plot.plot(self.episodes, self.score_values, color='b', label='score')
        self.score_plot.plot(self.episodes, self.mean_scores, color='g', label='mean')
        plt.draw()
        plt.pause(0.001)


    def write_log_header(self):
        f = open(self.logfile, 'w')       
        f.write('[TYPE OF LOG];Episodes;Loss;Score;Mean Score \n')
        f.close()

    def log_data(self):
        f = open(self.logfile, 'a')
        f.write("[DATA];{};{};{};{}\n".format(self.episodes[-1], self.loss_values[-1], self.score_values[-1], self.mean_scores[-1]))
        f.close()

    def load_training_data(self):
        f = open(self.logfile, 'r')
        next(f)
        for line in f:
            fields = line.split(';')
            if fields[0] == '[DATA]':
                self.episodes.append(int(fields[1]))
                self.loss_values.append(float(fields[2]))
                self.score_values.append(float(fields[3]))
                self.mean_scores.append(float(fields[4]))

        
    def visualize_data(self, episode, loss, score):
        self.episodes.append(episode)
        self.loss_values.append(loss)
        self.score_values.append(score)
        self.mean_scores.append(np.mean(self.score_values[-self.mean_subset:]))

        
        if episode % self.episode_ticks == 0:
            self.plot_score()
            self.plot_train_loss()
            

            print("Mean Score : {}, total_steps: {}".format(np.mean(self.score_values[-self.mean_subset:]), np.sum(self.score_values)))
            if episode > 0:
                self.log_data()
    def save_plot(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        plt.savefig(path + name)

    def reset(self):
        self.__init__(episode_ticks=self.episode_ticks, mean_subset=self.mean_subset)
