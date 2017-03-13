import numpy as np

def choice_n_rnd_numbers_from_to_linspace(start, end, space, n, integer=False, round=True, round_n=2, seed=None):
    """
    return n numbers choiced from a range created with np.linspace
    from start to end spaced as specified in space [there will be space numbers in range]
    :param start:
    :param end:
    :param space:
    :param n:
    :return:
    """

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    r1 = np.linspace(start, end, space)
    #keys = np.random.randint(0, len(r1), n)

    #choices = r1[keys]
    choices = np.random.choice(r1, n, replace=False)
    if integer:
        choices = choices.astype(int)

    if not integer:
        if round:
            choices = np.round(choices, round_n)

    return choices


if __name__ == '__main__':

    print choice_n_rnd_numbers_from_to_linspace(-10, 11, 100, 4, seed=110, round=True, round_n=4)
    print choice_n_rnd_numbers_from_to_linspace(-.5, .5, 100, 10, round=True)
    print choice_n_rnd_numbers_from_to_linspace(10, 100, 100, 3, integer=True)
