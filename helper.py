def periodic_padding(x, padding=1):
    '''
    x: shape (batch_size, d1, d2)
    return x padded with periodic boundaries. i.e. torus or donut
    '''
    d1 = x.shape[1] # dimension 1: height
    d2 = x.shape[2] # dimension 2: width
    p = padding
    # assemble padded x from slices
    #            tl,tc,tr
    # padded_x = ml,mc,mr
    #            bl,bc,br
    top_left = x[:, -p:, -p:] # top left
    top_center = x[:, -p:, :] # top center
    top_right = x[:, -p:, :p] # top right
    middle_left = x[:, :, -p:] # middle left
    middle_center = x # middle center
    middle_right = x[:, :, :p] # middle right
    bottom_left = x[:, :p, -p:] # bottom left
    bottom_center = x[:, :p, :] # bottom center
    bottom_right = x[:, :p, :p] # bottom right
    top = tf.concat([top_left, top_center, top_right], axis=2)
    middle = tf.concat([middle_left, middle_center, middle_right], axis=2)
    bottom = tf.concat([bottom_left, bottom_center, bottom_right], axis=2)
    padded_x = tf.concat([top, middle, bottom], axis=1)
    return padded_x