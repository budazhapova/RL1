#ifndef BANDIT
#define BANDIT

#include <random>
#include <iostream>

class Bandit {

	// constants for the experiments
	constexpr static int arms = 5;
	constexpr static int selections = 2000;
	constexpr static int runs = 1000;

	// normally distributed rewards (first problem)
	std::normal_distribution<double> normal_rewards [arms];
	int best_normal_arm = 0;

	// bernoulli distributions of probability of reward (second problem)
	std::bernoulli_distribution binary_rewards [arms];
	int best_binary_arm = 0;

	// performance statistics for all algorithms
	double avg_reward_eps_greedy[2][selections] = {0};
	double prc_optimal_eps_greedy[2][selections] = {0};

	public:
        // reset problems/arms
		void reset();

        // make greedy choice from array
        static int greedyChoice(const double *array);

        // reinforcement learning algorithms
		void epsilonGreedy(double epsilon);

		inline void bestArms() const{
		    std::cout << "best normal arm: " << best_normal_arm << std::endl
		              << "best binary arm: " << best_binary_arm << std::endl;
		}

		void outputResults() const;
};

#endif