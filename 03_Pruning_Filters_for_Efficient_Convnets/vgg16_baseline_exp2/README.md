# baseline developmenet

* learning rate : 0.1
* preprocessing : horizontal flip, rgb2yuv, mean-std, normalization
* Weights of conv layers are initialized MSR-style[3], known as He Initialization
* not use lr_scheduler
``` py
if epoch % epoch_step == 0 :
    optimizer.param_groups[0]['lr'] /= 2
```