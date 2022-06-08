def fwd_pass(X, y, train=False):
# IMPORTANTE, TRAIN = FALSE PARA QUE NO ENTRENE CON EL TEST DATA ESTO ES PARA PODER HACER TEST MIENTRAS ENTRENO Y VALIDO, 
# SE ESPERA QUE LA EXACTITUD EN EL TEST DE VALIDACIÓN SEA MENOR
    if train: 
        net.zero_grad()
        
    # NORMALIZATION
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    for i, x in enumerate(X):
        X[i] = normalize(X[i]/255) # Np array

    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
    y_pred = [torch.argmin(i) for i in outputs.cpu()] # 1 means positive diagnosis: (1,0) => 1
    y_true = [torch.argmin(i) for i in y.cpu()]
    conf_m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)


    """
    for i, x in enumerate(X):
        plt.xlabel(f"Label: {y_true[i]} Predicted: {y_pred[i]} Output: {outputs[i].cpu().detach().numpy()}")
        plt.imshow(X[i].permute(2, 1, 0).cpu())
        plt.show()
    print(conf_m)
    """
    
    if train:
        loss.backward() # Calculate gradients using backprop
        optimizer.step() # Updates W and b using previously calculated gradients

    return acc, loss, conf_m

def test():
  global test_set, test_dataloader, MODEL_NAME, MODE

  with open(r"C:\Users\Alejandro\Desktop\heterogeneous-data\results\WSI\test.csv", MODE) as f:
    acc, loss = 0, 0
    conf_m = np.array([[0,0],[0,0]])
    for batch_X, batch_y in tqdm(iter(test_dataloader)):
      batch_X, batch_y = batch_X.type(torch.FloatTensor).to(device).permute(0, 3, 2, 1), batch_y.type(torch.FloatTensor).to(device)

      net.eval() # Making sure that the model is not training and deactivate droptout

      with torch.no_grad(): # Disable all computations, works together with net.eval()
          acc_aux, loss_aux, conf_m_aux = fwd_pass(batch_X, batch_y, train=False)

      acc += acc_aux*(len(batch_X)/len(test_set)) # Calculating the average loss and acc trough batches
      loss += loss_aux*(len(batch_X)/len(test_set))
      conf_m += conf_m_aux

    prc = conf_m[0][0]/(conf_m[0][0]+conf_m[0][1])
    rec = conf_m[0][0]/(conf_m[0][0]+conf_m[0][1])
    f1 = 2*prc*rec/(prc+rec)

    print("Test loss: ", loss.item(), "\n")
    print("Test acc: ", acc, "\n")
    print("Test PRC: ", prc, "\n") # TP/TP+FP
    print("Test REC: ", rec, "\n") # TP/TP+FN
    print("f1: ", f1, "\n")
    print("\nCONF: \n", conf_m, "\n")

def train():
  global net, loss_function, scheduler, optimizer, train_set, val_set, MODEL_NAME, EPOCHS
  
  print(MODEL_NAME)

  with open(f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\results\\WSI\\log\\model_{MODEL_NAME}.log", MODE) as f:
    for epoch in range(EPOCHS):
      acc, loss = 0, 0,
      val_acc, val_loss = 0, 0,
      conf_m, val_conf_m = np.array([[0,0],[0,0]]), np.array([[0,0],[0,0]])

      print("\nEPOCH: ", epoch+1)

      for batch_X, batch_y in tqdm(iter(train_dataloader)):

        batch_X, batch_y = batch_X.type(torch.FloatTensor).to(device).permute(0, 3, 2, 1), batch_y.type(torch.FloatTensor).to(device) 
        
        net.train() # Making sure that the model is in training mode
        
        acc_aux, loss_aux, conf_m_aux = fwd_pass(batch_X, batch_y, train=True)
        
        acc += acc_aux*(len(batch_X)/len(train_set)) # Calculating the average loss and acc through batches sum ACCi*Wi/N (Wi = weight of the batch)
        loss += loss_aux*(len(batch_X)/len(train_set))
        conf_m += conf_m_aux

        """
        i += 1
    
        if i%100 == 0:
          print("Memory allocated in GPU: ", torch.cuda.memory_allocated("cuda:0")/1024/1024/1024)
        """
        
      for batch_X, batch_y in tqdm(iter(val_dataloader)):

        batch_X, batch_y = batch_X.type(torch.FloatTensor).to(device).permute(0, 3, 2, 1), batch_y.type(torch.FloatTensor).to(device)

        net.eval() # Making sure that the model is not training and deactivate droptout
        
        with torch.no_grad(): # Disable all computations, works together with net.eval()
          acc_aux, loss_aux, conf_m_aux = fwd_pass(batch_X, batch_y, train=False)

        val_acc += acc_aux*(len(batch_X)/len(val_set)) # Calculating the average loss and acc trough batches
        val_loss += loss_aux*(len(batch_X)/len(val_set))
        val_conf_m += conf_m_aux

      acc = (conf_m[0][0]+conf_m[1][1])/(conf_m[1][0]+conf_m[0][1]+conf_m[1][1]+conf_m[0][0]) # Better way to obtain acc tan using per batch acc
      val_acc = (val_conf_m[0][0]+val_conf_m[1][1])/(val_conf_m[1][0]+val_conf_m[0][1]+val_conf_m[1][1]+val_conf_m[0][0])

      prc = conf_m[1][1]/(conf_m[1][1]+conf_m[0][1])
      val_prc = val_conf_m[1][1]/(val_conf_m[1][1]+val_conf_m[0][1])
      rec = conf_m[1][1]/(conf_m[1][1]+conf_m[1][0])
      val_rec = val_conf_m[1][1]/(val_conf_m[1][1]+val_conf_m[1][0])
      f1 = 2*prc*rec/(prc+rec)
      val_f1 = 2*val_prc*val_rec/(val_prc+val_rec)
      
      print("Val loss: ", val_loss.item()," Train loss: ", loss.item(), "\n")
      print("Val acc: ", val_acc," Train acc: ", acc, "\n")
      print("Val PRC:", val_prc, "Train PRC: ", prc) # TP/TP+FP
      print("Val REC: ", val_rec,"Train REC: ", rec) # TP/TP+FN
      print("Val f1: ", val_f1," Train f1: ", f1, "\n")
      print("Val CONF: \n", val_conf_m,"\nTrain CONF: \n", conf_m, "\n")

      conf_m = f"{conf_m[0][0]}+{conf_m[0][1]}+{conf_m[1][0]}+{conf_m[1][1]}"
      val_conf_m = f"{val_conf_m[0][0]}+{val_conf_m[0][1]}+{val_conf_m[1][0]}+{val_conf_m[1][1]}"
    
      f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),3)},{round(float(loss),4)},{conf_m},{round(float(prc),4)},{round(float(rec),4)},")
      f.write(f"{round(float(val_acc),3)},{round(float(val_loss),4)},{val_conf_m}, {round(float(val_prc),4)}, {round(float(val_rec),4)}\n")
      f.write("\n\n")

      print("Learning Rate: ", optimizer.param_groups[0]["lr"])
      scheduler.step() # Changing the learning rate

    torch.save(net, f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\results\\WSI\\models\\{MODEL_NAME}.pth")

