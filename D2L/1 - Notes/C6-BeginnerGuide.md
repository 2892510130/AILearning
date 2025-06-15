## Chapter 6: Beginner Guide
- Tied layer: gradient will add up along different chain
- Custom initialization: `apply` method
- I/O
  - save tensor: `torch.save(x:Uinon[List[tensor], Dict], name:str)` and load
  - save model: the same, just input dict of the net (`net.state_dict()`) then `net.load_state_dict(torch.load(name))`
- GPU
  - operation between tensors must in the same GPU
  - print or transform to numpy will copy to memory, and even worse wait the python **GIL** (`Global Interpreter Lock`, make sure at the same time only one thread can execute the python bytecode)