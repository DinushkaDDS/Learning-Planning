import numpy as np
import random

TYPES_OF_ELECTRONICS = {0:"P1",
                        1:"P2", 
                        2:"P3", 
                        3:"P4", 
                        4:"P5"}

NUM_OF_SIMULATIONS = 1000
NUM_PRODUCTS = 5


def greedy_approach(TRUE_PRODUCT_SALES_PROBABILITIES):

    TOTAL_SALES = []
    # PRODUCT_SALES = np.zeros(NUM_PRODUCTS) # True product sales

    PRODUCT_ESTIMATED_REWARDS = np.zeros(NUM_PRODUCTS)
    PRODUCT_RECOMMENDED_COUNTS = np.zeros(NUM_PRODUCTS)

    for sample in range(NUM_OF_SIMULATIONS):

        # Select product with highest estimated probability of purchase
        if np.max(PRODUCT_RECOMMENDED_COUNTS) == 0:  # if all counts are zero, select randomly
            selected_product = np.random.randint(NUM_PRODUCTS)
        else:
            selected_product = np.argmax(PRODUCT_ESTIMATED_REWARDS)
        
        # Simulate purchase based on the true probability
        purchase_reward = np.random.rand() < TRUE_PRODUCT_SALES_PROBABILITIES[selected_product]
        
        # Update counts and estimates
        PRODUCT_RECOMMENDED_COUNTS[selected_product] += 1
        PRODUCT_ESTIMATED_REWARDS[selected_product] += (purchase_reward - PRODUCT_ESTIMATED_REWARDS[selected_product]) / PRODUCT_RECOMMENDED_COUNTS[selected_product]

        # Increment total sales if a purchase was made
        if purchase_reward:
            TOTAL_SALES.append(purchase_reward)

    return sum(TOTAL_SALES)
    

def epsilon_greedy(TRUE_PRODUCT_SALES_PROBABILITIES):
    TOTAL_SALES = []
    # PRODUCT_SALES = np.zeros(NUM_PRODUCTS) # True product sales

    PRODUCT_ESTIMATED_REWARDS = np.zeros(NUM_PRODUCTS)
    PRODUCT_RECOMMENDED_COUNTS = np.zeros(NUM_PRODUCTS)

    EPSILON = 0.1

    for sample in range(NUM_OF_SIMULATIONS):

        if np.max(PRODUCT_RECOMMENDED_COUNTS) == 0:  # if all counts are zero, select randomly
            selected_product = np.random.randint(NUM_PRODUCTS)
        else:
            selected_product = np.argmax(PRODUCT_ESTIMATED_REWARDS)

            select_best_or_not = np.random.rand() < (1 - EPSILON)
            if not (select_best_or_not):
                random_product = np.random.randint(NUM_PRODUCTS)
                while(random_product==selected_product):
                    selected_product = np.random.randint(NUM_PRODUCTS)
        
        # Simulate purchase based on the true probability
        purchase_reward = np.random.rand() < TRUE_PRODUCT_SALES_PROBABILITIES[selected_product]
        
        # Update counts and estimates
        PRODUCT_RECOMMENDED_COUNTS[selected_product] += 1
        PRODUCT_ESTIMATED_REWARDS[selected_product] += (purchase_reward - PRODUCT_ESTIMATED_REWARDS[selected_product]) / PRODUCT_RECOMMENDED_COUNTS[selected_product]

        # Increment total sales if a purchase was made
        if purchase_reward:
            TOTAL_SALES.append(purchase_reward)

    return sum(TOTAL_SALES)


def softmax_sampling(TRUE_PRODUCT_SALES_PROBABILITIES):

    TOTAL_SALES = []
    # PRODUCT_SALES = np.zeros(NUM_PRODUCTS) # True product sales

    PRODUCT_ESTIMATED_REWARDS = np.zeros(NUM_PRODUCTS)
    PRODUCT_RECOMMENDED_COUNTS = np.zeros(NUM_PRODUCTS)


    for sample in range(NUM_OF_SIMULATIONS):

        # Select product with highest probability of purchase
        softmax_probs = np.exp(PRODUCT_ESTIMATED_REWARDS) / np.sum(np.exp(PRODUCT_ESTIMATED_REWARDS))
        selected_product = np.random.choice(NUM_PRODUCTS, p=softmax_probs)
        
        # Simulate purchase based on the true probability
        purchase_reward = np.random.rand() < TRUE_PRODUCT_SALES_PROBABILITIES[selected_product]
        
        # Update counts and estimates
        PRODUCT_RECOMMENDED_COUNTS[selected_product] += 1
        PRODUCT_ESTIMATED_REWARDS[selected_product] += (purchase_reward - PRODUCT_ESTIMATED_REWARDS[selected_product]) / PRODUCT_RECOMMENDED_COUNTS[selected_product]

        # Increment total sales if a purchase was made
        if purchase_reward:
            TOTAL_SALES.append(purchase_reward)

    return sum(TOTAL_SALES)


def thompson_sampling(TRUE_PRODUCT_SALES_PROBABILITIES):
    TOTAL_SALES = []
    # PRODUCT_SALES = np.zeros(NUM_PRODUCTS) # True product sales

    ALPHA = np.ones(NUM_PRODUCTS)
    BETA = np.ones(NUM_PRODUCTS)

    for sample in range(NUM_OF_SIMULATIONS):

        # Select product with highest probability of purchase
        sampled_vals_for_each_arm = np.random.beta(ALPHA, BETA)
        selected_product = np.argmax(sampled_vals_for_each_arm)

        # Simulate purchase based on the true probability
        purchase_reward = np.random.rand() < TRUE_PRODUCT_SALES_PROBABILITIES[selected_product]
        
        # Increment total sales if a purchase was made
        if purchase_reward:
            TOTAL_SALES.append(purchase_reward)
            ALPHA[selected_product] += 1
        else:
            BETA[selected_product] += 1

    return sum(TOTAL_SALES)


def upper_confidence_bound_sampling(TRUE_PRODUCT_SALES_PROBABILITIES, C = 0.8):

    TOTAL_SALES = []
    # PRODUCT_SALES = np.zeros(NUM_PRODUCTS) # True product sales

    PRODUCT_RECOMMENDED_COUNTS = np.zeros(NUM_PRODUCTS)
    PRODUCT_ESTIMATED_REWARDS = np.zeros(NUM_PRODUCTS)

    for sample_num in range(NUM_OF_SIMULATIONS):

        ucb_values = np.zeros(NUM_PRODUCTS)
        for i in range(NUM_PRODUCTS):
            ucb_values[i] = PRODUCT_ESTIMATED_REWARDS[i] + C*np.sqrt((np.log(sample_num + 1)) / (PRODUCT_RECOMMENDED_COUNTS[i] + 1e-8))
        
        selected_product = np.argmax(ucb_values)

        # Simulate purchase based on the true probability
        purchase_reward = np.random.rand() < TRUE_PRODUCT_SALES_PROBABILITIES[selected_product]
        
        # Update counts and estimates
        PRODUCT_RECOMMENDED_COUNTS[selected_product] += 1
        PRODUCT_ESTIMATED_REWARDS[selected_product] += (purchase_reward - PRODUCT_ESTIMATED_REWARDS[selected_product]) / PRODUCT_RECOMMENDED_COUNTS[selected_product]

        # Increment total sales if a purchase was made
        if purchase_reward:
            TOTAL_SALES.append(purchase_reward)

    return sum(TOTAL_SALES)





























