
import pandas as pd

latencies = {
    "perlmutter": {
        "all_gather": 0.009,
        "reduce_scatter": 0.009,
    },
    "frontier":{
        "all_gather": 0.039,
        "reduce_scatter": 0.0298
    }
}

bws = {
    "perlmutter": {
        "all_gather": 59.52,
        "reduce_scatter": 59.52,
    },
    "frontier":{
        "all_gather": 109.98,
        "reduce_scatter": 127.79
    }
}



def get_bw(ip, my, machine="perlmutter", version="v2"):
    if version == "v1":
        return 1

    if machine == "perlmutter":
        if ip*my <=4:
            if ip==1:
                if my==2:
                    return 76
                elif my==4:
                    return 225
            
            elif ip == 2:
                if my == 2:
                    return 76
        return 59.52 / (min(4, ip))    
        
    elif machine == "frontier":
        if ip*my <= 8:
            if ip==1:
                if my==2:
                    return 129
                elif my==4:
                    return 52
                elif my==8:
                    return 135
            elif ip == 2:
                if my == 2:
                    return 50
                elif my == 4:
                    return 72
            elif ip == 4:
                if my == 2:
                    return 36 
        return 109.98 / (min(8, ip))

def model_v2(B, S, H1, H2, Gd, Gr, Gc, Gdata, machine="perlmutter", order=["c", "r", "d"], ignore_latency=False):
    dp_comm = 4 * (Gdata-1) / Gdata * H1 * H2 / (Gc*Gr*Gd) * 2 /1024/1024/1024
    depth_tensor_comm = 3/2 * 2 * (Gd-1) / Gd * H1 * H2 / (Gc*Gr) * 2 /1024/1024/1024
    row_tensor_comm = 2 * (Gr-1) / Gr * (B/Gdata/Gd * S * H1/Gc ) * 2 /1024/1024/1024
    col_tensor_comm = 2 * (Gc-1) / Gc * (B/Gdata/Gd * S * H2/Gr ) * 2 /1024/1024/1024

    col_time = row_time = depth_time = data_time = 0 
    

    ip=1
    latency_factor = 0 if ignore_latency else latencies[machine]["all_gather"]
    if Gc > 1:
        col_bw = get_bw(ip, Gc, machine)
        if col_bw is None:
            return None
        col_time = latency_factor * Gc / 1000  + col_tensor_comm / col_bw 
    ip*= Gc 
    if Gr > 1:
        row_bw = get_bw(ip, Gr, machine)
        if row_bw is None:
            return None
        row_time = latency_factor * Gr / 1000 + row_tensor_comm / row_bw 

    ip *= Gr 
    
    if Gd > 1:
        depth_bw = get_bw(ip, Gd, machine)
        if depth_bw is None:
            return None
        depth_time = latency_factor * Gd / 1000 +  depth_tensor_comm / depth_bw 

    ip *= Gd

    if Gdata > 1: 
        data_bw = get_bw(ip, Gdata, machine)
        if data_bw is None:
            return None
        data_time = latency_factor * Gdata / 1000 +  dp_comm / data_bw 


    return col_time + row_time + depth_time + data_time



def get_configs_for_transformer(
        global_batch_size_in_samples,
        sequence_length,
        num_layers,
        hidden_size,
        intermediate_size,
        swiglu, 
        GPUs,
        minimum_degree_of_tensor_parallelism,
        model_version="v2",
        topk=5,
        no_dp=False,
        machine="perlmutter",
        limit=None,
        gqa_factor=1,
        ignore_latency=False
):
    S=sequence_length
    K=3
    B=global_batch_size_in_samples
    H=hidden_size
    G=GPUs
    min_tp=minimum_degree_of_tensor_parallelism


    range = []
    i=0
    while 2**i <=G:
        range.append(2**i)
        i+=1

    data = {}
    ct = 0
    for Gc in range:
        for Gr in range:
            for Gd in range:
                for Gdata in range:
                    if Gc*Gr*Gd*Gdata == G and Gc*Gr*Gd>=min_tp and B%(Gdata*Gd)==0 and (not no_dp or Gdata==1):
                        #data[(Gc,Gr,Gd,Gdata)] =
                        ct += 1
                        # the 4 fc layers of a transformer,
                        # I swap Gc and Gr for "transposed layers" 
                        if limit is not None:
                            if Gc>limit or Gr>limit or Gd>limit or Gdata>limit:
                                continue

                        a = model_v2(B, S, H, H + 2* H // gqa_factor, Gd, Gr, Gc, Gdata, machine, ignore_latency=ignore_latency) 
                        b = model_v2(B, S, H, H, Gd, Gc, Gr, Gdata, machine, ignore_latency=ignore_latency)
                        c = model_v2(B, S, H, intermediate_size, Gd, Gr, Gc, Gdata, machine, ignore_latency=ignore_latency)
                        d = model_v2(B, S, H, intermediate_size, Gd, Gc, Gr, Gdata, machine, ignore_latency=ignore_latency)
                        if swiglu:
                            c *= 2

                        if a is None or b is None or c is None or d is None:
                            continue
                        else:
                            data[(Gc,Gr,Gd,Gdata)] = a + b + c + d
                        
    print("Total configs = ", ct)              
    sorted_configs = sorted(data.items(), key=lambda x:x[1])

    
    keys = "Gr","Gc","Gd","Gdata", "Comm-Time(s)"#,"Total-Time(s)"
    data = []
    for (Gc, Gr, Gd, Gdata), comm_time in sorted_configs[:topk]:
        comm_time = comm_time * num_layers
        data.append([Gr, Gc, Gd, Gdata, comm_time]), #total_time])

    df = pd.DataFrame(data, columns=keys)
    return df


# llama-2 7b
num_layer, hidden_size, intermediate_size, swiglu, min_tp, gqa_factor = 32, 4096, 11008, True, 8, 1 

# llama-2 70b
#num_layer, hidden_size, intermediate_size, swiglu, min_tp, gqa_factor = 80, 8192, 28672, True, 128, 10 



all = []
for g in [8]:
    df = get_configs_for_transformer(
        global_batch_size_in_samples=16,
        sequence_length=1,
        num_layers=num_layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        swiglu=swiglu, 
        GPUs=g,
        gqa_factor=gqa_factor,
        minimum_degree_of_tensor_parallelism=min_tp, 
        model_version="v2", #v1 or v2, although v1 won't work now, we need to fix it
        topk=-1, 
        no_dp=False, # if this is True, then all configs will have dp=1
        machine="perlmutter", # frontier or perlmutter
        limit=None,
        ignore_latency=False,
    )
    print(df)
    