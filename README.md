## Quaternion CNN

This repository contains code for [insert paper when uploaded].

All quaternions should be represented as tensors of shape `(4, 1)`, with the real part in the zeroth index. A tensor quaternion thought of as having shape `dims`, for example, should therefore actually have shape `(*dims, 4, 1)`.

The implementation of the quaternion kernels are in `qcnn.py`. The main exports of interest are:
* `qcnn.QConv1d` Quaternion convolutional kernel, accepts arguments `inchannels, outchannels, filterlen, stride=1` and tensors of shape `(batch, in channels, in time, 4, 1)`.
* `qcnn.QBatchNorm1d` Quaternion batch norm, accepts arguments `*dims, momentum=0.1` and tensors of shape `(batch, *dims, time, 4, 1)`.
* `qcnn.cuda()` Call this once to prepare the kernel to run on GPU.

Some utility exports of interest are
* `qcnn.checkGrad()` Call this to check that the implementations of quaternion gradients are correct.
* `qcnn.checkEquivariant()` Call this to check that the quaternion kernel is equivariant.
* `qcnn.qconj(q)` Returns the conjugate of `q`.
* `qcnn.qnormsq(q)` Returns the squared norm of `q`.
* `qcnn.qnorm(q)` Returns the norm of `q`.
* `qcnn.qinv(q)` Returns the inverse of `q`.
* `qcnn.rotate(q, r)` Rotates quaternion `q` by rotation quaternion `r`.

Example usages can be found in `models.py`, which defines a CNN and QCNN in parallel to illustrate the similar usages of the `nn.Conv1D`, `qcnn.QConv1d`, `nn.BatchNorm1D`, and `qcnn.QBatchNorm1d`. These were also the models used for the multi-user experiments in the paper.