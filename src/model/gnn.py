import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv
from torch_geometric.data import Data, Batch
from torch_scatter import scatter


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1, edge_feature=True):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        if edge_feature:
            self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        else:
            self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, dropout=dropout))       
        
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            if edge_feature:
                self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            else:
                self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if edge_feature:
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        else:
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, dropout=dropout,))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr
    
class QAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1, num_graph_token=8, edge_feature=True, gnn='gt'):
        super(QAG, self).__init__()
        
        if gnn not in ['gcn', 'gat', 'gt']:
            raise NotImplementedError('gnn must be one of value in GraphTransformer(gt), GAT(gat) and GCN(gcn)')
        elif gnn == 'gt':
            if num_heads == -1:
                raise ValueError("num_heads must be postive integer!")
            self.question_node_message_model = GraphTransformer(in_channels=in_channels,
                                                                hidden_channels=hidden_channels,
                                                                out_channels=hidden_channels,
                                                                dropout=dropout,
                                                                num_layers=1,
                                                                num_heads=num_heads,
                                                                edge_feature=False
                                                                )
            self.node_graphtoken_message_model = GraphTransformer(in_channels=out_channels,
                                                                hidden_channels=hidden_channels,
                                                                out_channels=out_channels,
                                                                dropout=dropout,
                                                                num_layers=1,
                                                                num_heads=num_heads,
                                                                edge_feature=False
                                                                )

            self.node_node_message_model = GraphTransformer(in_channels=hidden_channels,
                                                            hidden_channels=hidden_channels,
                                                            out_channels=out_channels,
                                                            dropout=dropout,
                                                            num_layers=num_layers,
                                                            num_heads=num_heads,
                                                            edge_feature=edge_feature
                                                            )
        elif gnn == 'gat':
            if num_heads == -1:
                raise ValueError("num_heads must be postive integer!")
            self.question_node_message_model = GAT(in_channels=in_channels,
                                                    hidden_channels=hidden_channels,
                                                    out_channels=hidden_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads
                                                    )
            self.node_graphtoken_message_model = GAT(in_channels=out_channels,
                                                    hidden_channels=hidden_channels,
                                                    out_channels=out_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads
                                                    )

            self.node_node_message_model = GAT(in_channels=hidden_channels,
                                                hidden_channels=hidden_channels,
                                                out_channels=out_channels,
                                                dropout=dropout,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                )
        elif gnn == 'gcn':
            self.question_node_message_model = GCN(in_channels=in_channels,
                                                    hidden_channels=hidden_channels,
                                                    out_channels=hidden_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads
                                                    )
            self.node_graphtoken_message_model = GCN(in_channels=out_channels,
                                                    hidden_channels=hidden_channels,
                                                    out_channels=out_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads
                                                    )

            self.node_node_message_model = GCN(in_channels=hidden_channels,
                                                hidden_channels=hidden_channels,
                                                out_channels=out_channels,
                                                dropout=dropout,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                )
        
        self.num_graph_token = num_graph_token
        self.graph_token = torch.nn.Parameter(torch.randn(self.num_graph_token, hidden_channels), requires_grad=True)
        
        self.query_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels),
        )
    
    def text_aware_graph_embedding(self, pyg_graph_list, text_embedds):
        device = next(self.parameters()).device
        text_embedding_tensor = text_embedds.to(device)
        
        text_embedding_tensor = self.query_node_mlp(text_embedding_tensor)
        
        num_graph = len(pyg_graph_list['ptr'])-1
        x_list = []
        edge_list = []
        for i in range(num_graph):
            x_list.append(pyg_graph_list.x[pyg_graph_list['batch']==i])
            x_list[-1] = torch.concat([text_embedding_tensor[i].unsqueeze(0), x_list[-1]], dim=0)
            
            cur_edge = [[0]*(len(x_list[-1])-1), list(range(1,len(x_list[-1])))]
            edge_list.append(torch.tensor(cur_edge, dtype=torch.long))

        res = []
        for i in range(len(x_list)):
            res.append(Data(x=x_list[i], edge_index=edge_list[i]))
        
        return Batch.from_data_list(res)
    
    def make_graphtoken_graph(self, pyg_graph_list):
        device = next(self.parameters()).device
        
        # To add text node
        num_graph = len(pyg_graph_list['ptr'])-1
        x_list = []
        edge_list = []
        for i in range(num_graph):
            x_list.append(pyg_graph_list.x[pyg_graph_list['batch']==i])
            x_list[-1] = torch.concat([self.graph_token.reshape(self.num_graph_token,-1), x_list[-1]], dim=0)
            
            edge1 = []
            edge2 = []
            for i in range(self.num_graph_token):
                edge1.extend([i]*(len(x_list[-1])-self.num_graph_token))
                edge2.extend(list(range(self.num_graph_token,len(x_list[-1]))))
            edge_list.append(torch.tensor([edge2, edge1], dtype=torch.long))

        res = []
        for i in range(len(x_list)):
            res.append(Data(x=x_list[i], edge_index=edge_list[i]))
        
        return Batch.from_data_list(res).to(device)
        
    def forward(self, pyg_graph_list, text_embedds=None, query_aware=True, pooling='graph_token'):
        device = next(self.parameters()).device
        num_graph = len(pyg_graph_list['ptr'])-1
        num_node_list = []
        for i in range(num_graph):
            num_node_list.append(sum(pyg_graph_list['batch']==i))
                
        if query_aware:
            if text_embedds == None:
                raise ValueError("query_aware is True but text_embedding_tensor is None.")
            
            
            #print("Input Graphs", pyg_graph_list)
            
            text_aware_batch = self.text_aware_graph_embedding(pyg_graph_list, text_embedds=text_embedds)
            text_aware_batch = text_aware_batch.to(device)
            #Query-Node Message Passing
            node_embedding, _ = self.question_node_message_model(x=text_aware_batch.x,
                                                                adj_t=text_aware_batch.edge_index,
                                                                edge_attr=None)

            node_embedding = node_embedding + text_aware_batch.x
            #print("Query aware node embedding with query embedding", node_embedding.size())
            #node_embedding = torch.nn.functional.relu(self.question_node_mlp(node_embedding))
            
            cur_start = 0
            temp = []
            for num in num_node_list:
                #print(cur_start+1, cur_start+num+1)
                temp.extend(node_embedding[cur_start+1:cur_start+num+1])
                cur_start += num+1
            node_embedding = torch.stack(temp)
            #print("Query aware node embedding", node_embedding.size())
            
            #Node-Node Message Passing
            node_embedding, _ = self.node_node_message_model(x=node_embedding, 
                                                            adj_t=pyg_graph_list.edge_index,
                                                            edge_attr=pyg_graph_list.edge_attr)
            #node_embedding = self.node_node_mlp(node_embedding)
            
            pyg_graph_list.x = node_embedding + pyg_graph_list.x
            
            #print("node-node embedding", node_embedding.size())
        else:
            node_embedding, _ = self.node_node_message_model(x=pyg_graph_list.x, 
                                                            adj_t=pyg_graph_list.edge_index,
                                                            edge_attr=pyg_graph_list.edge_attr)
            
            pyg_graph_list.x = node_embedding
            
        if pooling == 'graph_token':
            graph_token_batch = self.make_graphtoken_graph(pyg_graph_list)
            
            graph_token_embedding, _ = self.node_graphtoken_message_model(x=graph_token_batch.x,
                                                                        adj_t=graph_token_batch.edge_index,
                                                                        edge_attr=None)
            
            #print("graph token embedding with node embedding", graph_token_embedding.size())
            
            #graph_token_embedding = torch.nn.functional.relu(self.node_graph_mlp(graph_token_embedding))       
            
            res_graph_token_embedding = []
            res_node_embedding = []
            cur_start=0
            for num in num_node_list:
                #print(cur_start+self.num_graph_token, cur_start+num+self.num_graph_token)
                res_graph_token_embedding.append(graph_token_embedding[cur_start:cur_start+self.num_graph_token])
                res_node_embedding.append(graph_token_embedding[cur_start+self.num_graph_token:cur_start+num+self.num_graph_token])
                cur_start += num+self.num_graph_token
            
            res_graph_token_embedding = torch.stack(res_graph_token_embedding)
            res_graph_token_embedding = res_graph_token_embedding
            #print(res_graph_token_embedding.size())
            #print(node_embedding.size())
            
            #return res_graph_token_embedding, node_embedding
            return res_graph_token_embedding, ""
        elif pooling == 'sum':
            #return torch_geometric.nn.global_add_pool(pyg_graph_list.x. pyg_graph_list.batch), node_embedding
            #return torch_geometric.nn.global_add_pool(pyg_graph_list.x, pyg_graph_list.batch).reshape(num_graph,1,-1)
            prev = pyg_graph_list['ptr'][0]
            zero_node_list=[]
            for i, ptr in enumerate(pyg_graph_list['ptr'][1:]):
                if ptr == prev:
                    zero_node_list.append(i)
                prev = ptr
            pooled_output = scatter(pyg_graph_list.x, pyg_graph_list.batch, dim=0, reduce='mean')
            embedding_list = []
            cur = 0
            if len(zero_node_list) != 0 :
                for i in range(num_graph):
                    if i in zero_node_list:
                        embedding_list.append(torch.zeros_like(pooled_output[0]))
                    else:
                        embedding_list.append(pooled_output[cur])
                        cur += 1
                pooled_output = torch.stack(embedding_list).reshape(num_graph,1,-1)
            else:
                pooled_output = pooled_output.reshape(num_graph,1,-1)
            return scatter(pyg_graph_list.x, pyg_graph_list.batch, dim=0, reduce='sum').reshape(num_graph,1,-1), ""
        elif pooling == 'mean':
            #return torch_geometric.nn.global_mean_pool(pyg_graph_list.x. pyg_graph_list.batch), node_embedding
            #return torch_geometric.nn.global_mean_pool(pyg_graph_list.x, pyg_graph_list.batch).reshape(num_graph,1,-1)

            prev = pyg_graph_list['ptr'][0]
            zero_node_list=[]
            for i, ptr in enumerate(pyg_graph_list['ptr'][1:]):
                if ptr == prev:
                    zero_node_list.append(i)
                prev = ptr
            pooled_output = scatter(pyg_graph_list.x, pyg_graph_list.batch, dim=0, reduce='mean')
            embedding_list = []
            cur = 0
            if len(zero_node_list) != 0 :
                for i in range(num_graph):
                    if i in zero_node_list:
                        embedding_list.append(torch.zeros_like(pooled_output[0]))
                    else:
                        embedding_list.append(pooled_output[cur])
                        cur += 1
                pooled_output = torch.stack(embedding_list).reshape(num_graph,1,-1)
            else:
                pooled_output = pooled_output.reshape(num_graph,1,-1)
            return pooled_output, ""
        else:
            raise ValueError('Pooling must takes one of the values "graph_token", "sum" and "mean".')
    
class CrossQAG(torch.nn.Module):
    def __init__(self, tokenizer, llm, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1, num_graph_token=8, edge_feature=True, gnn='gt', num_ealry_fusion_layer=2):
        super(CrossQAG, self).__init__()
        
        self.tokenizer = tokenizer
        self.llm = llm
        self.llm_hidden_dim = self.llm.config.hidden_size
        self.word_embedding = self.llm.model.get_input_embeddings()
        
        self.early_fusion_node_projection_layer = torch.nn.Linear(in_channels, self.llm_hidden_dim, dtype=torch.bfloat16)
        self.early_fusion_edge_projection_layer = torch.nn.Linear(in_channels, self.llm_hidden_dim, dtype=torch.bfloat16)
        
        self.early_fusion_mlp_layer_list = []
        self.early_fusion_attn_layer_list = []
        for _ in range(num_ealry_fusion_layer):
            early_fusion_attn_layer = torch.nn.MultiheadAttention(self.llm_hidden_dim, num_heads, batch_first=True, dtype=torch.bfloat16).to(self.llm.device)
            early_fustion_mlp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim, dtype=torch.bfloat16),
                torch.nn.Dropout(p=dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim, dtype=torch.bfloat16),
            ).to(self.llm.device)
            
            self.early_fusion_mlp_layer_list.append(early_fustion_mlp_layer)
            self.early_fusion_attn_layer_list.append(early_fusion_attn_layer)
        
        self.early_fusion_mlp_layer_list = torch.nn.ModuleList(self.early_fusion_mlp_layer_list)
        self.early_fusion_attn_layer_list = torch.nn.ModuleList(self.early_fusion_attn_layer_list)
    

        if gnn not in ['gcn', 'gat', 'gt']:
            raise NotImplementedError('gnn must be one of value in GraphTransformer(gt), GAT(gat) and GCN(gcn)')
        elif gnn == 'gt':
            if num_heads == -1:
                raise ValueError("num_heads must be postive integer!")
            self.node_node_message_model = GraphTransformer(in_channels=self.llm_hidden_dim,
                                                            hidden_channels=self.llm_hidden_dim,
                                                            out_channels=out_channels,
                                                            dropout=dropout,
                                                            num_layers=num_layers,
                                                            num_heads=num_heads,
                                                            edge_feature=edge_feature
                                                            )
            self.node_graphtoken_message_model = GraphTransformer(in_channels=out_channels,
                                                    hidden_channels=self.llm_hidden_dim,
                                                    out_channels=out_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads,
                                                    edge_feature=False
                                                    )
           
        elif gnn == 'gat':
            self.node_node_message_model = GAT(in_channels=self.llm_hidden_dim,
                                                hidden_channels=self.llm_hidden_dim,
                                                out_channels=out_channels,
                                                dropout=dropout,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                )
            self.node_graphtoken_message_model = GAT(in_channels=out_channels,
                                                hidden_channels=hidden_channels,
                                                out_channels=out_channels,
                                                dropout=dropout,
                                                num_layers=1,
                                                num_heads=num_heads
                                                )
        elif gnn == 'gcn':
            self.node_node_message_model = GCN(in_channels=self.llm_hidden_dim,
                                                hidden_channels=self.llm_hidden_dim,
                                                out_channels=out_channels,
                                                dropout=dropout,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                )
            self.node_graphtoken_message_model = GCN(in_channels=out_channels,
                                                    hidden_channels=hidden_channels,
                                                    out_channels=out_channels,
                                                    dropout=dropout,
                                                    num_layers=1,
                                                    num_heads=num_heads
                                                    )
        self.node_node_message_model = self.node_node_message_model.type(torch.bfloat16)
        self.node_graphtoken_message_model = self.node_graphtoken_message_model.type(torch.bfloat16) 
    
        self.num_graph_token = num_graph_token
        self.graph_token = torch.nn.Parameter(torch.randn(self.num_graph_token, out_channels, dtype=torch.bfloat16), requires_grad=True) 
    
    def get_word_embedding(self, input_text_list):
        word_embedding_list = []
        input_ids_list = self.tokenizer(input_text_list, add_special_tokens=False)
        
        for input_ids in input_ids_list['input_ids']:
            word_embedding = self.word_embedding(torch.tensor(input_ids).to(self.llm.device))
            word_embedding_list.append(word_embedding)
        
        return word_embedding_list
    
    def make_graphtoken_graph(self, pyg_graph_list):
        device = next(self.parameters()).device
        
        # To add text node
        num_graph = len(pyg_graph_list['ptr'])-1
        x_list = []
        edge_list = []
        for i in range(num_graph):
            x_list.append(pyg_graph_list.x[pyg_graph_list['batch']==i])
            x_list[-1] = torch.concat([self.graph_token.reshape(self.num_graph_token,-1), x_list[-1]], dim=0)
            
            edge1 = []
            edge2 = []
            for i in range(self.num_graph_token):
                edge1.extend([i]*(len(x_list[-1])-self.num_graph_token))
                edge2.extend(list(range(self.num_graph_token,len(x_list[-1]))))
            edge_list.append(torch.tensor([edge2, edge1], dtype=torch.long))

        res = []
        for i in range(len(x_list)):
            res.append(Data(x=x_list[i], edge_index=edge_list[i]))
        
        return Batch.from_data_list(res).to(device)
    
    def forward(self, samples, pooling='graph_token'):
        samples['graph'].x = self.early_fusion_node_projection_layer(samples['graph'].x.type(torch.bfloat16))
        samples['graph'].edge_attr = self.early_fusion_edge_projection_layer(samples['graph'].edge_attr.type(torch.bfloat16))
        device = next(self.parameters()).device
        num_graph = len(samples['graph']['ptr'])-1
        batch_size = len(samples['id'])
        input_text_list = [samples["desc"][i]+samples["question"][i] for i in range(batch_size)]
        num_node_list = []
        for i in range(num_graph):
            num_node_list.append(sum(samples['graph']['batch']==i))
        
        word_embedding_list = self.get_word_embedding(input_text_list)
        
        query_aware_x_list = []
        for i in range(num_graph):
            x = samples['graph'].x[samples['graph']['batch']==i].to(device)
            word_embedding = word_embedding_list[i]
            for i in range(len(self.early_fusion_mlp_layer_list)):
                x, _ = self.early_fusion_attn_layer_list[i](x, word_embedding, word_embedding)
                x = self.early_fusion_mlp_layer_list[i](x)
            query_aware_x_list.append(x)
            
        samples['graph'].x = torch.cat(query_aware_x_list, dim=0)
        
        node_embedding, _ = self.node_node_message_model(x=samples['graph'].x,
                                                            adj_t=samples['graph'].edge_index,
                                                            edge_attr=samples['graph'].edge_attr)
        
        samples['graph'].x = node_embedding
        
        if pooling == 'graph_token':
            graph_token_batch = self.make_graphtoken_graph(samples['graph'])
            
            graph_token_embedding, _ = self.node_graphtoken_message_model(x=graph_token_batch.x,
                                                                        adj_t=graph_token_batch.edge_index,
                                                                        edge_attr=None)
            #print("graph token embedding with node embedding", graph_token_embedding.size())
            
            #graph_token_embedding = torch.nn.functional.relu(self.node_graph_mlp(graph_token_embedding))       
            
            res_graph_token_embedding = []
            res_node_embedding = []
            cur_start=0
            for num in num_node_list:
                #print(cur_start+self.num_graph_token, cur_start+num+self.num_graph_token)
                res_graph_token_embedding.append(graph_token_embedding[cur_start:cur_start+self.num_graph_token])
                res_node_embedding.append(graph_token_embedding[cur_start+self.num_graph_token:cur_start+num+self.num_graph_token])
                cur_start += num+self.num_graph_token
            
            res_graph_token_embedding = torch.stack(res_graph_token_embedding)

            return res_graph_token_embedding, ""
        elif pooling == 'sum':
            #return torch_geometric.nn.global_add_pool(pyg_graph_list.x. pyg_graph_list.batch), node_embedding
            #return torch_geometric.nn.global_add_pool(pyg_graph_list.x, pyg_graph_list.batch).reshape(num_graph,1,-1)
            prev = samples['graph']['ptr'][0]
            zero_node_list=[]
            for i, ptr in enumerate(samples['graph']['ptr'][1:]):
                if ptr == prev:
                    zero_node_list.append(i)
                prev = ptr
            pooled_output = scatter(samples['graph'].x, samples['graph'].batch, dim=0, reduce='mean')
            embedding_list = []
            cur = 0
            if len(zero_node_list) != 0 :
                for i in range(num_graph):
                    if i in zero_node_list:
                        embedding_list.append(torch.zeros_like(pooled_output[0]))
                    else:
                        embedding_list.append(pooled_output[cur])
                        cur += 1
                pooled_output = torch.stack(embedding_list).reshape(num_graph,1,-1)
            else:
                pooled_output = pooled_output.reshape(num_graph,1,-1)
            return scatter(samples['graph'].x, samples['graph'].batch, dim=0, reduce='sum').reshape(num_graph,1,-1), ""
        elif pooling == 'mean':
            #return torch_geometric.nn.global_mean_pool(pyg_graph_list.x. pyg_graph_list.batch), node_embedding
            #return torch_geometric.nn.global_mean_pool(pyg_graph_list.x, pyg_graph_list.batch).reshape(num_graph,1,-1)

            prev = samples['graph']['ptr'][0]
            zero_node_list=[]
            for i, ptr in enumerate(samples['graph']['ptr'][1:]):
                if ptr == prev:
                    zero_node_list.append(i)
                prev = ptr
            pooled_output = scatter(samples['graph'].x, samples['graph'].batch, dim=0, reduce='mean')
            embedding_list = []
            cur = 0
            if len(zero_node_list) != 0 :
                for i in range(num_graph):
                    if i in zero_node_list:
                        embedding_list.append(torch.zeros_like(pooled_output[0]))
                    else:
                        embedding_list.append(pooled_output[cur])
                        cur += 1
                pooled_output = torch.stack(embedding_list).reshape(num_graph,1,-1)
            else:
                pooled_output = pooled_output.reshape(num_graph,1,-1)
            return pooled_output, ""
        else:
            raise ValueError('Pooling must takes one of the values "graph_token", "sum" and "mean".')
            

load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
    'qag':QAG,
    'cross_qag':CrossQAG
}
