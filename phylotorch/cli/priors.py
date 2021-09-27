def create_one_on_x_prior(id_, theta):
    return {
        'id': id_,
        'type': 'Distribution',
        'distribution': 'OneOnX',
        'x': theta,
    }
