import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RLEvaluation:
    def __init__(self, episode_ticks = 1, mean_subset = 100):
        self.episodes, self.loss_values, self.score_values, self.mean_scores = [], [], [], []
        self.episode_ticks = episode_ticks
        self.mean_subset = mean_subset

        #init plots
        sns.set(style="whitegrid")
        plt.ion()
        self.loss_plot = plt.subplot(211)
        self.score_plot = plt.subplot(212)
        self.loss_plot.set_ylabel("loss values")
        self.loss_plot.set_xlabel("episodes")
        self.score_plot.set_ylabel("score values")
        self.score_plot.set_xlabel("episodes")
        plt.show(block=False)

    def plot_train_loss(self):
        self.loss_plot.semilogy(self.episodes, self.loss_values)
        plt.draw()
        plt.pause(0.001)

    def plot_score(self):
        self.score_plot.plot(self.episodes, self.score_values, color='b', label='score')
        self.score_plot.plot(self.episodes, self.mean_scores, color='g', label='mean')
        plt.draw()
        plt.pause(0.001)

    def visualize_data(self, episode, loss, score):
        self.episodes.append(episode)
        self.loss_values.append(loss)
        self.score_values.append(score)
        self.mean_scores.append(np.mean(self.score_values[-self.mean_subset:]))


        if episode % self.episode_ticks == 0:
            self.plot_score()
            self.plot_train_loss()

            print("Mean Score : {}, total_steps: {}".format(np.mean(self.score_values[-self.mean_subset:]), np.sum(self.score_values)))
