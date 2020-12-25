time_synchronized()函数里面进行了torch.cuda.synchronize()，再返回的time.time()
torch.cuda.synchronize()等待gpu上完成所有的工作
总的来说就是这样测试时间会更准确