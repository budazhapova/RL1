#include <fstream>
#include <algorithm>
#include <cmath>
#include "Bandit.h"

void Bandit::reset(){
    // generator to use with distributions
    std::random_device device;
    std::default_random_engine generator(device());

    // get means of reward distributions from normal distribution
    std::normal_distribution<double> mean_dist(0, 1);
    // get probability of reward == 1 for binary arms from uniform distribution
    std::uniform_real_distribution<double> prob_dist(0, 1);

    // set reward distributions for all normal arms
    for (int arm = 0; arm < arms; arm++) {
        double reward_mean = mean_dist(generator);
        normal_rewards[arm] = std::normal_distribution<double>(reward_mean, 1);
        double reward_prob = prob_dist(generator);
        binary_rewards[arm] = std::bernoulli_distribution(reward_prob);
    }

    // find best choices for normal problem arms and binary problem arms
    best_normal_arm = 0;
    best_binary_arm = 0;
    for (int arm = 1; arm < arms; arm++) {
        if (normal_rewards[arm].mean() > normal_rewards[best_normal_arm].mean())
            best_normal_arm = arm;
        if (binary_rewards[arm].p() > binary_rewards[best_binary_arm].p())
            best_binary_arm = arm;
    }
}

int Bandit::greedyChoice(const double *array){
    // pick first value as highest arbitrarily, then compare to find actual highest value
    double highest_value = array[0];
    std::vector<int> best_options = {0};

    for (int choice = 1; choice < arms; choice++) {
        double choice_value = array[choice];
        if (choice_value > highest_value) {
            best_options = {choice};
            highest_value = choice_value;
        }
        else if (choice_value == highest_value)
            best_options.push_back(choice);
    }
    if (best_options.size() == 1)
        return best_options[0];

    // if there are multiple choices tied, return one of them randomly
    auto generator = std::default_random_engine(std::random_device{}());
    int tiebreaker = std::uniform_int_distribution<int>(0, best_options.size() - 1)(generator);
    return best_options[tiebreaker];
}

void Bandit::epsilonGreedy(double epsilon){
    // use <random> distributions to use for choosing greedily with a probability
    // and to determine the random choice
    std::random_device device;
    std::default_random_engine generator(device());
    std::bernoulli_distribution p_random(epsilon);
    std::uniform_int_distribution<int> random_choice(0, arms - 1);

    for (int n = 0; n < runs; n++){
        // reset reward distributions at start of each run
        this->reset();
        // keep track of estimated rewards (Q) and times actions were taken
        double Q [2][arms] = {0};
        int times_taken [2][arms] = {0};

        for (int t = 0; t < selections; t++){
            // has a chance of epsilon to choose randomly, otherwise greedy
            int normal_selection, binary_selection;
            // and reward received at current step
            double reward [2];
            // and whether the optimal choice was made
            bool optimal_choice [2];

            if (p_random(generator)){
                normal_selection = random_choice(generator);
                binary_selection = random_choice(generator);
            } else {
                normal_selection = greedyChoice(Q[0]);
                binary_selection = greedyChoice(Q[1]);
            }

            reward[0] = normal_rewards[normal_selection](generator);
            times_taken[0][normal_selection]++;
            optimal_choice[0] = (normal_selection == best_normal_arm);

            reward[1] = binary_rewards[binary_selection](generator);
            times_taken[1][binary_selection]++;
            optimal_choice[1] = (binary_selection == best_binary_arm);

            // update estimated rewards Q
            Q[0][normal_selection] += (1.0 / (times_taken[0][normal_selection])) *
                                       (reward[0] - Q[0][normal_selection]);
            Q[1][binary_selection] += (1.0 / (times_taken[1][binary_selection])) *
                                        (reward[1] - Q[1][binary_selection]);

            // keep track of averages and optimal choices
            avg_reward[0][0][t] += reward[0] * 1.0 / runs;
            avg_reward[0][1][t] += reward[1] * 1.0 / runs;
            if (optimal_choice[0])
                prc_optimal[0][0][t] += 100.0 / runs;
            if (optimal_choice[1])
                prc_optimal[0][1][t] += 100.0 / runs;
        }
    }
}

void Bandit::optimisticInitValues(double alpha, double init){
    // sets initial estimation to init and updates estimates with the rate alpha
    std::random_device device;
    std::default_random_engine generator(device());

    for (int n = 0; n < runs; n++){
        // same setup reboot as in epsilonGreedy (see above for explanation)
        this->reset();
        // for optimistic algorithm the initial estimates are (equally) high
        double Q [2][arms];
        for (int q = 0; q < 2; q ++)
            std::fill_n (Q[q], arms, init);
        int times_taken [2][arms] = {0};

        for (int t = 0; t < selections; t++){
            int normal_selection, binary_selection;
            double reward [2];
            bool optimal_choice [2];

            // always chooses greedily, but starts with high expectations
            normal_selection = greedyChoice(Q[0]);
            binary_selection = greedyChoice(Q[1]);

            reward[0] = normal_rewards[normal_selection](generator);
            times_taken[0][normal_selection]++;
            optimal_choice[0] = (normal_selection == best_normal_arm);

            reward[1] = binary_rewards[binary_selection](generator);
            times_taken[1][binary_selection]++;
            optimal_choice[1] = (binary_selection == best_binary_arm);

            // update estimated rewards Q
            Q[0][normal_selection] += alpha * (reward[0] - Q[0][normal_selection]);
            Q[1][binary_selection] += alpha * (reward[1] - Q[1][binary_selection]);

            // keep track of averages and optimal choices
            avg_reward[1][0][t] += reward[0] * 1.0 / runs;
            avg_reward[1][1][t] += reward[1] * 1.0 / runs;
            if (optimal_choice[0])
                prc_optimal[1][0][t] += 100.0 / runs;
            if (optimal_choice[1])
                prc_optimal[1][1][t] += 100.0 / runs;
        }
    }
}

void Bandit::reinforcementComparison(double alpha){
    // random generator to use for action selection later
    std::default_random_engine generator;

    for (int n = 0; n < runs; n++){
        this->reset();
        // reference reward
        double ref_reward[2] = {0};
        // keep track of preferences for and probabilities of taking various actions
        double preference[2][arms] = {0};
        // double probability[2][arms] = {0};
        std::array<double, arms> probability[2];

        for (int t = 0; t < selections; t++){
            double sum_of_probs[2] = {0};
            int choice [2];
            double reward [2];
            bool optimal_choice [2];

            // get the sum of all action probabilities for the denominator (see below)
            for (int i = 0; i < arms; i++){
                sum_of_probs[0] += exp(preference[0][i]);
                sum_of_probs[1] += exp(preference[1][i]);
            }
            // compute new probabilities of choosing every action
            for (int a = 0; a < arms; a++){
                probability[0][a] = exp(preference[0][a]) / sum_of_probs[0];
                probability[1][a] = exp(preference[1][a]) / sum_of_probs[1];
            }

            // generate probability distributions for all actions using the computed probabilities
            std::discrete_distribution<int> normal_selection(probability[0].begin(), probability[0].end());
            std::discrete_distribution<int> binary_selection(probability[1].begin(), probability[1].end());
            // action selection and getting the reward
            choice[0] = normal_selection(generator);
            choice[1] = binary_selection(generator);
            reward[0] = normal_rewards[choice[0]](generator);
            reward[1] = binary_rewards[choice[1]](generator);
            // check whether it was the optimal action
            optimal_choice[0] = (choice[0] == best_normal_arm);
            optimal_choice[1] = (choice[1] == best_binary_arm);

            // update the reference reward
            ref_reward[0] += alpha * (reward[0] - ref_reward[0]);
            ref_reward[1] += alpha * (reward[1] - ref_reward[1]);
            // update the preference of the chosen action
            preference[0][choice[0]] += reward[0] - ref_reward[0];
            preference[1][choice[1]] += reward[1] - ref_reward[1];

            // keep track of averages and optimal choices
            avg_reward[2][0][t] += reward[0] * 1.0 / runs;
            avg_reward[2][1][t] += reward[1] * 1.0 / runs;
            if (optimal_choice[0])
                prc_optimal[2][0][t] += 100.0 / runs;
            if (optimal_choice[1])
                prc_optimal[2][1][t] += 100.0 / runs;
        }
    }
}

void Bandit::outputResults() const
{
    std::string algorithm_names[3] = {"eps_greedy", "optimistic", "reinforcement_comp"}; // NB! expand later
    std::ofstream file;
    file.open("results.csv");
    file << "bandit" << ',' << "selection" << ',' << "algorithm" << ','
         << "average_reward" << ',' << "optimal_percentage" << '\n';
    for (int bandit = 0; bandit < 2; bandit++)
        for (int selection = 0; selection < selections; selection++)
            for (int algorithm = 0; algorithm < 3; algorithm++) // NB! revise stop condition
            {
                file << (bandit == 0 ? "normal_dist" : "bernoulli_dist") << ','
                     << selection << ',' << algorithm_names[algorithm] << ','
                     /*<< avg_reward_eps_greedy[bandit][selection] << ','
                     << prc_optimal_eps_greedy[bandit][selection] << '\n';*/
                     << avg_reward[algorithm][bandit][selection] << ','
                     << prc_optimal[algorithm][bandit][selection] << '\n';
            }
    file.close();
}
