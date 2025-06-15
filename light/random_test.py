import numpy as np

def reandom_test():
    rating_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    user_id = np.array([3, 1, 2, 0, 1, 2])
    item_id = np.array([0, 3, 2, 5, 4, 0])
    print(rating_matrix)
    print([user_id, item_id])
    print(rating_matrix[user_id, item_id])

if __name__ == "__main__":
    reandom_test()