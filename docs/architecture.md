### Why ConvLSTM

Standard image classifiers evaluate each frame independently, which throws away exactly the information that matters most for activity recognition: how motion evolves from one frame to the next. A single frame of someone raising an arm looks identical whether it's a wave or the start of a punch — the context is in the sequence, not the snapshot.

3D-CNNs solve part of this by convolving across time as well as space, but they need large amounts of labeled data and heavy compute to learn long-range temporal patterns well. Two-stream networks (combining raw RGB frames with a separate optical-flow stream) capture motion more explicitly, but computing optical flow for every frame is expensive and hard to run anywhere close to real-time. ConvLSTM was the better fit here: it folds convolutional filters directly into an LSTM's recurrent cells, so spatial feature extraction and temporal memory happen in the same layer, in a single forward pass, without a separate motion-estimation step.

### Model Architecture

The network takes 20-frame sequences (64×64×3 per frame) and passes them through four stacked ConvLSTM2D layers with progressively increasing filter counts — 4, 8, 14, then 16 — so each layer builds on slightly deeper spatial features than the last. A MaxPooling3D layer follows each ConvLSTM block to shrink the spatial dimensions while preserving the temporal axis, and TimeDistributed Dropout is applied after the first few blocks to regularize across every timestep rather than just the final output. The stack ends with a Flatten layer and a Dense(3, softmax) layer, mapping the learned spatiotemporal features to Normal / Violence / Weaponized. Trained with categorical cross-entropy, the Adam optimizer, and early stopping on validation loss, the model converged over 25 epochs to 91.63% test accuracy.

### A Real Constraint I Had to Design Around

Training ran on Kaggle's free-tier GPU, which has a hard memory ceiling — and ConvLSTM layers are memory-hungry because they carry hidden state across every one of the 20 timesteps per sequence, on top of the usual convolutional activations. Early architecture attempts with higher filter counts ran out of GPU memory mid-training. The fix wasn't a smarter model — it was accepting a leaner one: dropping filter counts to the 4→16 range and keeping batch size at 4 kept memory usage inside Kaggle's limit without sacrificing much accuracy. It's a good example of a real engineering trade-off: the "correct" model on paper isn't always the one you can actually train with the compute you have.
