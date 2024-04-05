import torch
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary 
from torch_lr_finder import LRFinder

class Utils:
    def __init__(self):
        pass

    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def show_random_images_for_each_class(self, train_data, num_images_per_class=16):
        for c, cls in enumerate(train_data.classes):
            rand_targets = random.sample([
                n
                for n, x in enumerate(train_data.targets)
                if x==c
            ], k=num_images_per_class)
            show_img_grid(
                np.transpose(train_data.data[rand_targets], axes=(0, 3, 1, 2))
            )
            plt.title(cls)
    

    def show_img_grid(self, data):
        try:
            grid_img = torchvision.utils.make_grid(data.cpu().detach())
        except:
            data = torch.from_numpy(data)
            grid_img = torchvision.utils.make_grid(data)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        

    def show_random_images(self, data_loader):
        data, target  = next(iter(data_loader))
        self.show_img_grid(data)


    def show_model_summary(self, model, input_size=(1, 28, 28)):
        summary(model, input_size=input_size)

    def find_lr(self, model, optimizer, criterion, device, trainloader, numiter, startlr, endlr):
        lr_finder = LRFinder(
            model=model, optimizer=optimizer, criterion=criterion, device=device
        )

        lr_finder.range_test(
            train_loader=trainloader,
            start_lr=startlr,
            end_lr=endlr,
            num_iter=numiter,
            step_mode="exp",
        )

        lr_finder.plot()

        lr_finder.reset()


    def one_cycle_lr(self, optimizer, maxlr, steps, epochs):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=maxlr,
            steps_per_epoch=steps,
            epochs=epochs,
            pct_start=5 / epochs,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return scheduler

    def training_plots(self, results):
        plt.plot(results["epoch"], results["trainloss"])
        plt.plot(results["epoch"], results["testloss"])
        plt.legend(["Train Loss", "Validation Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.show()

        plt.plot(results["epoch"], results["trainacc"])
        plt.plot(results["epoch"], results["testacc"])
        plt.legend(["Train Acc", "Validation Acc"])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.show()


    def lr_plots(self, results, length):
        plt.plot(range(length), results["lr"])
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate vs Epochs")
        plt.show()
