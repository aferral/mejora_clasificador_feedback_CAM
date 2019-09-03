# @do_profile()
def exp_oclusion_vs_CAM(dataset,classifier):

    dataset.initialize_iterator_val(classifier.sess)

    # Generate list of masks with noise pattern
    mask_dim = dataset.shape if len(dataset.shape) == 2 else dataset.shape[0:2]

    box = np.zeros((int(dataset.shape[0] * 0.2), int(dataset.shape[1] * 0.2),3))  # make a 5x5 grid of replaces
    lx = int(dataset.shape[0] / box.shape[0])
    ly = int(dataset.shape[1] / box.shape[1])

    dx = box.shape[0]
    dy = box.shape[1]

    all_pos = [[dx * i, dx * (i + 1), dy * j, dy * (j + 1)] for i in
               range(lx) for j in range(ly)]

    mask_images = np.zeros(tuple([len(all_pos)] + mask_dim))
    for i, t in enumerate(all_pos):
        mask_images[i, t[0]:t[1], t[2]:t[3]] = 1

    while True:
        with timeit():
            try:
                fd = classifier.prepare_feed(is_train=False, debug=True)

                # Calc prediction, target, conv_acts
                tensors=[classifier.last_conv, classifier.softmax_weights, classifier.targets, classifier.pred]
                conv_acts,softmax_w,y_real ,y_pred = classifier.sess.run(tensors, fd)

                pred_class = np.argmax(y_pred,axis=1)

                batch_s = conv_acts.shape[0]
                n_filters = softmax_w.shape[0]

                # Calc CAM of predictions
                # np.array_equal( softmax_w[:,pred_class][:,2], softmax_w[:,pred_class[2]])

                predicted_soft_w = softmax_w[:,pred_class]
                predicted_soft_w = predicted_soft_w.T.reshape(batch_s, 1, 1, n_filters) # (for broadcasting in the filter h,w dimension)

                cam_maps = (conv_acts * predicted_soft_w).sum(axis=-1) # Element wise multiplication per channel and then sum all the channels for a batch
                # equivalent to np.array_equal((conv_acts[1,:, :, :] * softmax_w[:, pred_class[1]]).sum(axis=2), res[1])

                # oclusion maps
                oclusion_maps = np.zeros([batch_s] + mask_dim)
                alt_oclusions = []

                im_batch = fd['model_input:0']

                # Apply all mask for each image in batch
                for ind,image in enumerate(im_batch):
                    new_batch = np.zeros(tuple([len(all_pos)] +  dataset.shape ))
                    current=0
                    for ind_pos in range(len(all_pos)):
                        t = all_pos[ind_pos]
                        new_batch[current] = image
                        new_batch[current,t[0]:t[1],t[2]:t[3]] = 0
                        current += 1

                    fd['model_input:0'] = new_batch
                    oc_pred = classifier.sess.run([classifier.pred], fd)[0][:,pred_class[ind]]

                    alt_o = oc_pred.reshape(5, 5)
                    alt_o = (alt_o - alt_o.min()) / (alt_o.max() - alt_o.min())
                    alt_o = (alt_o * 2) - 1
                    alt_o = invert(alt_o) - 1
                    alt_o = resize(alt_o, mask_dim)
                    alt_oclusions.append(alt_o)

                    # Add oclusion map as prob of class if that area is covered
                    oclusion_maps[ind] += (mask_images * oc_pred.reshape(len(all_pos),1,1)).sum(axis=0)

                print("BATCH DONE")

            except tf.errors.OutOfRangeError:
                break
