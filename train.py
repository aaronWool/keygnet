import torch
from torch.autograd import Variable
from torch import autograd
from tensorboardX import SummaryWriter
import open3d as o3d
import numpy as np

from model import KeyGNet



def cal_loss_dispersion(input,gamma):
    '''
    input: keypoint tensor [b,n,3]
            b-batch
            n-number of points
            3-x,y,z
    
    output: dispersion loss
    '''
    n = input.size()[1]
    iter = 0
    sum=0
    for kpoints in input:
        for i in range(0,n-1):
            for j in range(i+1,n):
                #if i!=j:
                #print("[",i,", ",j,"]")
                distance = ((kpoints[i,0]-kpoints[j,0])**2+(kpoints[i,1]-kpoints[j,1])**2+(kpoints[i,2]-kpoints[j,2])**2)**0.5
                sum += torch.exp(gamma*distance)
                #sum += distance
                iter += 1
    return sum/iter

def cal_loss_wass_gp(input,segments,vote_type,model,device,lambda_,training):
    '''
    input: keypoint tensor [b,n,3]
            b-batch
            n-number of points
            3-x,y,z
    segment: segment point cloud tensor [b,N,3]
            b-batch
            N-number of points
            3-x,y,z    
    vote_type: 0-radii
               1-vector
               2-offset    
    return wass loss
    ''' 
    b = input.size()[0]   
    n = input.size()[1]
    N = segments.size()[2]
    sum = 0
    iter = 0
    for i in range(b):
        kpoints = input[i]
        segment = segments[i]
        segment = segment.permute(1,0)
        #if vote_type==0:
            #votes = torch.zeros(n,N)
            #for j in range(n):
                #kpoint=kpoints[j]
                #votes[j]=((kpoint[0]-segment[:,0])**2+(kpoint[1]-segment[:,1])**2+(kpoint[2]-segment[:,2])**2)**0.5
        #else:
        #if vote_type!=0:
        votes = torch.zeros(n,N,3).cuda(device) 
        norms = torch.zeros(n,N).cuda(device)
        sorted_norms = torch.zeros(n,N).cuda(device)
        sorted_indices = torch.zeros(n,N,dtype=torch.long).cuda(device)
        for j in range(n):
            kpoint=kpoints[j]
            #print(kpoint.shape)
            #print(segment.shape)
            un_squeezed_kpt = torch.unsqueeze(kpoint,0)
            un_squeezed_kpt = un_squeezed_kpt.repeat(N,1)
            #print(un_squeezed_kpt.shape)
            #print(segment[:,:3].shape)
            xyz=segment[:,:3]
            offsets=torch.sub(un_squeezed_kpt,xyz)
            norm=torch.norm(offsets,dim=1)
            #print(norm.shape)
            #print(norms.shape)
            norms[j]=norm
            #norms[j]=((kpoint[0]-segment[:,0])**2+(kpoint[1]-segment[:,1])**2+(kpoint[2]-segment[:,2])**2)**0.5
            sorted_norms[j],sorted_indices[j]=torch.sort(norms[j])
            if vote_type!=0:
                votes[j] = offsets
            if vote_type==1:
                #norm = ((kpoint[0]-segment[:,0])**2+(kpoint[1]-segment[:,1])**2+(kpoint[2]-segment[:,2])**2)**0.5
                repeated_norm = norm.repeat(3,1)
                repeated_norm = repeated_norm.permute(1,0)
                vectors = torch.div(offsets,repeated_norm)
                votes[j] = vectors
            if vote_type!=0:
                #cross product
                vote = votes[j]

                cross = torch.cross(vote,un_squeezed_kpt)
                cross_copy = cross.detach().clone()
                norms[j] = torch.norm(cross_copy,dim=1)
                sorted_norms[j],sorted_indices[j]=torch.sort(norms[j])
                #votes[j] = votes[j,sorted_indices[j],:]

        for j in range(0,n-1):
            for k in range(i+1,n): 
                # if vote_type==0:
                wass_dist=torch.mean(torch.abs(sorted_norms[i]-sorted_norms[k]))
                sum+=wass_dist
                # else:
                #     wass_dist=torch.mean(torch.abs(votes[i]-votes[k]))
                #     sum+=wass_dist
                iter+=1
    l_wass=sum/iter
    if training:
        idx=np.random.choice(np.arange(b),b,replace=False)
        shuffled = segments[idx,:,:]
        #print(shuffled.size())
        #gradient penalty
        shuffled = Variable(shuffled, requires_grad=True)

        prob_shuffled = model(shuffled)
        #print(prob_shuffled.get_device())

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_shuffled, inputs=shuffled,
                               grad_outputs=torch.ones(
                                   prob_shuffled.size()).cuda(device) if args.cuda else torch.ones(
                                   prob_shuffled.size()),
                               create_graph=True, retain_graph=True)[0]
        #print(gradients.get_device())

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
        #print('wass: ', l_wass)
        #print('gp: ', grad_penalty)
        l_wass+=grad_penalty
    return l_wass                  
            
                    
#print(loss_dispersion(torch.rand(1,5,3)))
#print(loss_wass_gp(torch.rand(2,3,3), torch.rand(2,1024,3),0))

if __name__ == "__main__":
    import os
    import argparse
    import tqdm
    import shutil
    import numpy as np
    from model import KeyGNet
    from data_loader import BOPDataset

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser(description='KeyGNet Model Test')
    parser.add_argument('--keypointsNo', type=int, default=3, metavar='keypointsNo',
                        help='No, of Keypoints')   
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of test batch)')                     
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--optim', type=str, default='Adam',choices=['Adam', 'SGD'],
                        help='optimizer for training') 
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables GPU training')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points forwarded into the network')
    parser.add_argument('--min_vis_points', type=int, default=2000,
                        help='threshold for minimum segment points') 
    parser.add_argument('--data_root', type=str, default='data/lm',
                        help='dataset root dir')   
    parser.add_argument('--ckpt_root', type=str, default='',
                        help='root dir for ckpt')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dgcnn dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='dgcnn dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='dgcnn random seed (default: 1)')
    parser.add_argument('--lambda_term', type=float, default=1,
                        help='gradient penalty lambda term')
    parser.add_argument('--gamma', type=float, default=-0.5,
                        help='gamma for dispersion loss')
    parser.add_argument('--vote_type', type=int, default=0, choices=[0,1,2],
                        help='vote type to train on. radii:0, vector:1, offset:2.')
    vote_types = {
            0:'radii',
            1:'vector',
            2:'offset'}
    args = parser.parse_args()
    dataset_name = args.data_root.split('/')[-1]
    log_dir = os.path.join('logs',dataset_name,vote_types[args.vote_type])
    print(log_dir)
    # create log root
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    if args.cuda and torch.cuda.is_available():
        print("Using GPU!")
        device = 'cuda:0'
        torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
        torch.manual_seed(args.seed)

    # initialize model, optimizer, and dataloader; load ckpt
    model = KeyGNet(args,device).to(device)
    if args.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    if args.ckpt_root != '':
        if os.path.isfile(args.ckpt_root):
            checkpoint = torch.load(args.ckpt_root)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.ckpt_root))
    
    tblogger = SummaryWriter(logdir=os.path.join(log_dir,'tblog'))

    model.train()
    train_loader = torch.utils.data.DataLoader(BOPDataset(args.data_root,'train_primesense',
                                                          min_visible_points=args.min_vis_points,
                                                          points_count_net = args.num_points),
                                               batch_size=args.batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(BOPDataset(args.data_root,'test',
                                                         min_visible_points=args.min_vis_points,
                                                         points_count_net = args.num_points),
                                              batch_size=args.test_batch_size,shuffle=False)
    train_iteration = 0
    test_iteration = 0
    best_test_loss = np.inf
    save_name = "ckpt.pth.tar"
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
    for epoch in range(args.epochs):
        if epoch<=50:
            alpha = .7
            beta = .3
        else:
            alpha = .3
            beta = .7           
        for batch_idx, pc in tqdm.tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc='Train epoch=%d' % epoch,
                ncols=80,
                leave=False):
            pc = torch.permute(pc, (0, 2, 1))
            pc = pc.to(device)
            #pc = torch.rand(32,6,1024).to(device)
            #print(pc.shape)
            training=True
            optim.zero_grad()
            est_kpts = model(pc)
            loss_dis = cal_loss_dispersion(est_kpts,args.gamma)
            #print(loss_dis)
            loss_wass = cal_loss_wass_gp(est_kpts,pc,args.vote_type,model,device,args.lambda_term,training)
            loss = alpha*loss_wass+beta*loss_dis
            loss.backward()
            optim.step()
            np_loss_dis, np_loss_wass, np_loss = loss_dis.detach().cpu().numpy(), loss_wass.detach().cpu().numpy(),loss.detach().cpu().numpy()
            tblogger.add_scalar('Train', np_loss, train_iteration)
            tblogger.add_scalar('Train_wass', np_loss_wass, train_iteration)
            tblogger.add_scalar('Train_disp', np_loss_dis, train_iteration)
            train_iteration+=1
        test_losses=[]
        #dump ckpt

        torch.save(
            {
                'epoch': epoch,
                'optim_state_dict': optim.state_dict(),
                'model_state_dict': model.state_dict(),
            }, os.path.join(log_dir, save_name))
        
        #test loop
        if epoch%5 == 0 and epoch != 0:
            with torch.no_grad():
                for batch_idx, pc in tqdm.tqdm(
                        enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test for epoch %d' % epoch,
                        ncols=80,
                        leave=False):
                    pc = torch.permute(pc, (0, 2, 1))
                    pc = pc.to(device)
                    est_kpts = model(pc)
                    loss_dis = cal_loss_dispersion(est_kpts,args.gamma)
                    training=False
                    loss_wass = cal_loss_wass_gp(est_kpts,pc,args.vote_type,model,device,args.lambda_term,training)
                    loss = alpha*loss_wass+beta*loss_dis
                    #loss.back_ward()
                    #optim.step()
                    np_loss_dis, np_loss_wass, np_loss = loss_dis.detach().cpu().numpy(), loss_wass.detach().cpu().numpy(),loss.detach().cpu().numpy()
                    tblogger.add_scalar('Test', np_loss, train_iteration)
                    tblogger.add_scalar('Test_wass', np_loss_wass, train_iteration)
                    tblogger.add_scalar('Test_disp', np_loss_dis, train_iteration)
                    test_iteration+=1 
                    test_losses.append(np_loss)
                test_loss = np.mean(np.array(test_losses))
                scheduler.step(test_loss)

                if test_loss<best_test_loss:
                    best_test_loss=test_loss
                    #replace best model
                    shutil.copy(os.path.join(log_dir, save_name),
                            os.path.join(log_dir, 'model_best.pth.tar'))
    print(".....Saving keypoints to file",os.path.join(log_dir,'keypoints.npy'),".....")
    #load best model
    checkpoint = torch.load(os.path.join(log_dir,'model_best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pts_list = []
    colors_list = []
    #load normalization parameters 
    coor_dims = np.loadtxt(os.path.join(args.data_root,'split','coor_dims.txt'))
    #load cad models and generate pc
    ## use models_reconst for tless
    cad_models_path = os.path.join(args.data_root,"models_reconst")
    for filename in os.listdir(cad_models_path):
        if filename.endswith(".ply"):
            cad_model = o3d.io.read_point_cloud(os.path.join(cad_models_path,filename))
            pts = np.asarray(cad_model.points)
            colors = np.asarray(cad_model.colors)
            #print(colors.shape)
            #cad_models.append(cad_model)
            idx=np.random.choice(np.arange(pts.shape[0]), args.num_points, replace=False)
            pts = pts[idx]
            for i in range(3):
                #print(i)
                pts[:,i] -= np.mean(pts[:,i])
                pts[:,i] /=coor_dims[i]
            colors = colors[idx]
            pts_list.append(pts)
            colors_list.append(colors)
    pts_list = np.array(pts_list)
    colors_list = np.array(colors_list)
    #concatenate color and pts 
    pts_list = np.concatenate((pts_list,colors_list),axis=2)
    pts_list = torch.from_numpy(pts_list).float().cuda(device)
    pts_list = torch.permute(pts_list, (0, 2, 1))
    print(pts_list.shape)
    est_kpts = model(pts_list)
    #unnormalize
    est_kpts = est_kpts.cpu().detach().numpy()
    est_kpts[:,:,0] *= coor_dims[0]
    est_kpts[:,:,1] *= coor_dims[1]
    est_kpts[:,:,2] *= coor_dims[2]
    np.save(os.path.join(log_dir,'keypoints.npy'),est_kpts)                  


        
