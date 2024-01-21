## [NeurIPS 2022] AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition

- [\[NeurIPS 2022\] AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition](#neurips-2022-adaptformer-adapting-vision-transformers-for-scalable-visual-recognition)
  - [Image Codes](#image-codes)
    - [main\_image.py](#main_imagepy)


Basic information could be seen in [README.md](README.md). This is a document for instruction of how to read the whole codes (for self-use in research).


### Image Codes
#### main_image.py


**def get_args_parser():** a function to create and return a parser for command-line arguments. A document for this function could be found [here](https://docs.python.org/3/library/argparse.html).

**def set_trainable_adapter(model, idx):** a function for *layer-wise iterative training*, to set the trainable adapter in each layer.

**def main(args):** 
- Set the log dictionary
  ```
  if args.log_dir is None:
        args.log_dir = args.output_dir
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    ```
- Build Dataset
- fine-tuning configs
  ```
  tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
        adpnum_option = args.adpnum_option,
    )
  ```
- Build model
  ```
  if args.model.startswith('vit'):
        model = vit_image.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        drop_path_rate=args.drop_path,
        tuning_config=tuning_config,
        )
    else:
        raise NotImplementedError(args.model)

  if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
  ```
- Set the model head
  ```
  model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)


  for name, p in model.named_parameters():
      if name in msg.missing_keys:
            p.requires_grad = True
      else:
            p.requires_grad = False if not args.fulltune else True
  for _, p in model.head.named_parameters():
      p.requires_grad = True
  ```
- Training
  ```
      print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    if args.adpnum_option == 'single':
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                max_norm=None,
                log_writer=log_writer,
                args=args
            )
            if args.output_dir:
                misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_str = f"Memory Usage: {memory_info.rss / (1024 * 1024)} MB"

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(memory_usage_str + "\n")

    elif args.adpnum_option == 'multi':
        for step in range(args.steps):
            for i in range(len(args.ffn_num)):
                # freeze all but the head
                for _, p in model.named_parameters():
                    p.required_grad = False

                for _, p in model.head.named_parameters():
                    p.requires_grad = True

                model.to(device)

                # set the adapter i trainable
                set_trainable_adapter(model, i)

                optimizer = torch.optim.SGD([p for name, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
                print(optimizer)

                for epoch in range(args.start_epoch, args.epochs):
                    if args.distributed:
                        data_loader_train.sampler.set_epoch(epoch)
                    train_stats = train_one_epoch(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        max_norm=None,
                        log_writer=log_writer,
                        args=args
                    )
                    
                    test_stats = evaluate(data_loader_val, model, device)
                    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                    max_accuracy = max(max_accuracy, test_stats["acc1"])
                    print(f'Max accuracy: {max_accuracy:.2f}%')

                    if log_writer is not None:
                        log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                        log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                        log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch,
                                'n_parameters': n_parameters}

                    if args.output_dir and misc.is_main_process():
                        if log_writer is not None:
                            log_writer.flush()
                        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_usage_str = f"Memory Usage: {memory_info.rss / (1024 * 1024)} MB"

                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(memory_usage_str + "\n")

                if args.output_dir:
                        misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=step * len(args.ffn_num) + i)

    else:
        raise ValueError(args.ffn_adapt_num)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Training time is.{}".format(total_time_str) + "\n")
  ```
  
#### vit_image.py

**class VisionTransformer(nn.Module):** A class for transformer. 

```
self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
self.norm = norm_layer(embed_dim)

# Classifier head(s)
self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
self.head_dist = None
if distilled:
    self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

def forward_features(self, x):
    B = x.shape[0]

    # embed
    x = self.patch_embed(x)

    # build token
    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    # blks
    for idx, blk in enumerate(self.blocks):
        if self.tuning_config.vpt_on:
            eee = self.embeddings[idx].expand(B, -1, -1)
            x = torch.cat([eee, x], dim=1)
        x = blk(x)
        if self.tuning_config.vpt_on:
            x = x[:, self.tuning_config.vpt_num:, :]

    # pooling
    if self.global_pool:
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        outcome = self.fc_norm(x)
    else:
        x = self.norm(x)
        outcome = x[:, 0]

    return outcome

def forward(self, x):
    x = self.forward_features(x,)
    if self.head_dist is not None:
        x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        if self.training and not torch.jit.is_scripting():
            # during inference, return the average of both classifier predictions
            return x, x_dist
        else:
            return (x + x_dist) / 2
    else:
        x = self.head(x)
    return x
```

**Some examples**: 

```
def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
```
