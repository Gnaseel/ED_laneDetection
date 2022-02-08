python main.py --mode inference --backbone ResNet18_delta_SCNN --device 1
python main.py --mode score --backbone ResNet18_delta_SCNN --device 1









        # inferencer.model = self.getModel().to(self.device)
        # inferencer.model.load_state_dict(torch.load(self.cfg.model_path, map_location='cpu'))
        # inferencer.model.eval()
        # if self.cfg.backbone=="ResNet34_deg"  or self.cfg.backbone=="ResNet18_delta_SCNN":
        #     print("SET Model 2")
        #     # inferencer.model2 = ResNet34_seg()
        #     inferencer.model2 = ResNet18_seg_SCNN()
        #     inferencer.model2.to(self.device)
        #     inferencer.model2.eval()

        #     heat_path = os.path.join(CFG.weight_file_path, CFG.heat_weight_file)
        #     inferencer.model2.load_state_dict(torch.load(heat_path, map_location='cpu'))
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/01_27_18_34_device_cuda:1/epoch_0_index_339.pth", map_location='cpu'))
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/01_27_13_27_device_cuda:1/epoch_42_index_339.pth", map_location='cpu'))
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/01_15_13_50_device_cuda:2/epoch_200_index_339.pth", map_location='cpu'))

        #     # By CULANE
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/01_12_00_29_device_cuda:2/epoch_20_index_468.pth", map_location='cpu'))
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/12_27_17_41_device_cuda:0/epoch_100_index_339.pth", map_location='cpu'))
        #     # inferencer.model2.load_state_dict(torch.load("/home/ubuntu/Hgnaseel_SHL/Network/weight_file/lane_segmentation/epoch_70_index_339.pth", map_location='cpu'))

        # inferencer.device = self.device
        # inferencer.model.to(self.device)