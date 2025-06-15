## Chapter 13 : Computation
- compiler
  - net = torch.jit.script(net)
- Automatic Parallesim
  - y.to('cpu', non_blocking=non_blocking) for y in x, will return x[i-1] when calculate x[i]
- Tranning on multiple GPU
  - <figure style="text-align: center;">
      <img alt="Partion Methods" src="https://d2l.ai/_images/splitting.svg" style="background-color: white; display: inline-block;">
      <figcaption> Partion Methods </figcaption>
    </figure>
  - nn.parallel.scatter to split data to different devices
  - 显式同步（torch.cuda.synchronize()）仅在需要精确测量执行时间或调试异步错误时必要，其他情况会自己根据cpu或者后续数据需求隐式调用
- Concise impl :
  - What we need to do
    - Network parameters need to be initialized across all devices.
    - While iterating over the dataset minibatches are to be divided across all devices.
    - We compute the loss and its gradient in parallel across devices.
    - Gradients are aggregated and parameters are updated accordingly.
  - Use torch.nn.parallel.DistributedDataParallel
- Parameter Server
  - <figure style="text-align: center;">
      <img alt="Parameter Exchange" src="https://d2l.ai/_images/ps-distributed.svg" style="background-color: white; display: inline-block;">
      <figcaption> Parameter Exchange </figcaption>
    </figure>
  - last graph above assume gradient can be divided into four parts, and exchange each one of them each GPU.
  - Ring Synchronization
  - Key–Value Stores

<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn.svg" style="background-color: white; display: inline-block;"> -->
<!-- <img alt="ResNeXt Block" src="https://d2l.ai/_images/rnn-bptt.svg" style="background-color: white; display: inline-block;"> -->