#ifndef BANDIT
#define BANDIT

#include <random>
#include <iostream>
#include <array>

class Bandit {

	// constants for the experiments
	constexpr static int arms = 10;
	constexpr static int selections = 10000;
	constexpr static int runs = 1000;

	// normally distributed rewards (first problem)
	std::normal_distribution<double> normal_rewards [arms];
	int best_normal_arm = 0;

	// bernoulli distributions of probability of reward (second problem)
	std::bernoulli_distribution binary_rewards [arms];
	int best_binary_arm = 0;

	// performance statistics for all algorithms
	double avg_reward[4][2][selections] = {0};
	double prc_optimal[4][2][selections] = {0};

	/* order of aggregate algorithms (first index of) the score arrays:
	0 -- epsilon greedy
	1 -- optimistic epsilon greedy
	2 -- reinforcement comparison
	3 -- upper confidence bound */

	// total reward statistics across 4 algorithms, 2 bandits, all runs
	double total_reward[4][2][runs] = {0};

	public:
        // reset problems/arms
		void reset();

        // make greedy choice from array
        static int greedyChoice(const double *array);
		// choose an arm in UCB
		int actionSelectionUCB(const double *Q, int t, int times_taken[], double c);

        // reinforcement learning algorithms
		void epsilonGreedy(double epsilon);
		void optimisticInitValues(double alpha, double init);
		void reinforcementComparison(double alpha);
		void upperConfidenceBound(double c);

		// for testing/debugging purposes
		inline void bestArms() const{
		    std::cout << "best normal arm: " << best_normal_arm << std::endl
		              << "best binary arm: " << best_binary_arm << std::endl;
		}

		void outputResults() const;
};

#endif