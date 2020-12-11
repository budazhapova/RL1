#include <fstream>
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
    // pick first value as highest arbitrarily, then find actual highest value
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
    return best_options[std::uniform_int_distribution<int>
            (0, best_options.size() - 1)(generator)];
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

        // and reward received at time t
        double reward [2][selections];
        // and whether the optimal choice was made
        bool optimal_choice [2][selections];

        for (int t = 0; t < selections; t++) {
            // has a chance of epsilon to choose randomly, otherwise greedy
            int normal_selection, binary_selection;

            if (p_random(generator)){
                normal_selection = random_choice(generator);
                binary_selection = random_choice(generator);
            } else {
                normal_selection = greedyChoice(Q[0]);
                binary_selection = greedyChoice(Q[1]);
            }

            reward[0][t] = normal_rewards[normal_selection](generator);
            times_taken[0][normal_selection]++;
            optimal_choice[0][t] = (normal_selection == best_normal_arm);

            reward[1][t] = binary_rewards[binary_selection](generator);
            times_taken[1][binary_selection]++;
            optimal_choice[1][t] = (binary_selection == best_binary_arm);

            // update estimated rewards Q
            Q[0][normal_selection] += (1.0 / (times_taken[0][normal_selection])) *
                                       (reward[0][t] - Q[0][normal_selection]);
            Q[1][binary_selection] += (1.0 / (times_taken[1][binary_selection])) *
                                        (reward[1][t] - Q[1][binary_selection]);

            // keep track of averages and optimal choices
            avg_reward_eps_greedy[0][t] += reward[0][t] * 1.0 / runs;
            avg_reward_eps_greedy[1][t] += reward[1][t] * 1.0 / runs;
            if (optimal_choice[0][t])
                prc_optimal_eps_greedy[0][t] += 100.0 / runs;
            if (optimal_choice[1][t])
                prc_optimal_eps_greedy[1][t] += 100.0 / runs;
        }
    }
}

void Bandit::outputResults() const
{
    std::string algorithm_names[1] = {"eps_greedy"};
    std::ofstream file;
    file.open("results.csv");
    file << "bandit" << ',' << "selection" << ',' << "algorithm" << ','
         << "average_reward" << ',' << "optimal_percentage" << '\n';
    for (int bandit = 0; bandit < 2; bandit++)
        for (int selection = 0; selection < selections; selection++)
            for (int algorithm = 0; algorithm != 1; algorithm++)
            {
                file << (bandit == 0 ? "normal_dist" : "bernoulli_dist") << ','
                     << selection << ',' << algorithm_names[algorithm] << ','
                     << avg_reward_eps_greedy[bandit][selection] << ','
                     << prc_optimal_eps_greedy[bandit][selection] << '\n';
            }
    file.close();
}
