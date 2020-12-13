#ifndef BANDIT
#define BANDIT

#include <random>
#include <iostream>
#include <array>

class Bandit {

	// constants for the experiments
	constexpr static int arms = 10;
	constexpr static int selections = 2000;
	constexpr static int runs = 1000;

	// normally distributed rewards (first problem)
	std::normal_distribution<double> normal_rewards [arms];
	int best_normal_arm = 0;

	// bernoulli distributions of probability of reward (second problem)
	std::bernoulli_distribution binary_rewards [arms];
	int best_binary_arm = 0;

	// performance statistics for all algorithms -- expand as more algorithms implemented!
	double avg_reward[3][2][selections] = {0};
	double prc_optimal[3][2][selections] = {0};
	/* order of aggregate algorithms in the arrays above:
	0 -- epsilon greedy
	1 -- optimistic epsilon greedy
	2 -- reinforcement comparison */

	public:
        // reset problems/arms
		void reset();

        // make greedy choice from array
        static int greedyChoice(const double *array);

        // reinforcement learning algorithms
		void epsilonGreedy(double epsilon);
		void optimisticInitValues(double alpha, double init);
		void reinforcementComparison(double alpha);

		inline void bestArms() const{
		    std::cout << "best normal arm: " << best_normal_arm << std::endl
		              << "best binary arm: " << best_binary_arm << std::endl;
		}

		void outputResults() const;
};

#endif