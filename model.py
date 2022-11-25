import torch
from torch import nn
from torch.utils.data import Dataset
import pennylane as qml


class SequenceDataset(Dataset):
    def __init__(self,train_data,target,multi=False):
        self.X = torch.tensor(train_data).float()
        self.y = torch.tensor(target).float()
        self.multi = multi

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,i):
        x = self.X[i]
        if self.multi == False:
            x = x[None,:]
        return x,self.y[i]


class QLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,n_qubits=4,n_qlayers=1,n_vrotations=3,batch_first=True,return_sequences=False,return_state=False,backend= "default.qubit"):
        super(QLSTM,self).__init__()
        self.n_inputs = input_size # size of the data input
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_update_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend,wires=self.wires_forget)
        self.dev_input = qml.device(self.backend,wires=self.wires_input)
        self.dev_update = qml.device(self.backend,wires=self.wires_update)
        self.dev_output = qml.device(self.backend,wires=self.wires_output)

        def ansatz(params, wires_type):
            for i in range(1,3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j],wires_type[j+i]])
                    else:
                        qml.CNOT(wires=[wires_type[j],wires_type[j+i-self.n_qubits]])

            for i in range(self.n_qubits):
                qml.RX(params[0][i],wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features,weights,wires_type):
            ry_params = [torch.arctan(feature) for feature in features]
            rz_params = [torch.arctan(feature**2) for feature in features]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i],wires=wires_type[i])
                qml.RZ(rz_params[i],wires=wires_type[i])
            qml.layer(ansatz,self.n_qlayers,weights,wires_type=wires_type)

        def _circuit_forget(inputs,weights):
            VQC(inputs,weights,self.wires_forget)
            return [ qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        def _circuit_input(inputs,weights):
            VQC(inputs,weights,self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        def _circuit_update(inputs,weights):
            VQC(inputs,weights,self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        def _circuit_output(inputs,weights):
            VQC(inputs,weights,self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_forget = qml.QNode(_circuit_forget,self.dev_forget,interface="torch")
        self.qlayer_input = qml.QNode(_circuit_input,self.dev_input,interface="torch")
        self.qlayer_update = qml.QNode(_circuit_update,self.dev_update,interface="torch")
        self.qlayer_output = qml.QNode(_circuit_output,self.dev_output,interface="torch")

        weight_shapes = {"weights":(self.n_qlayers,self.n_vrotations,self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size,self.n_qubits)
        self.VQC = {
            'forget':qml.qnn.TorchLayer(self.qlayer_forget,weight_shapes),
            'input':qml.qnn.TorchLayer(self.qlayer_input,weight_shapes),
            'update':qml.qnn.TorchLayer(self.qlayer_update,weight_shapes),
            'output':qml.qnn.TorchLayer(self.qlayer_output,weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self,x,init_states = None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        # print(x.size())
        if self.batch_first is True:
            batch_size,seq_length,feature_size = x.size()
        else:
            seq_length,batch_size,features_size = x.size()

        # print(batch_size,seq_length,feature_size)

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size,self.hidden_size)
            c_t = torch.zeros(batch_size,self.hidden_size)
            # print(h_t)
            # print(c_t)
        else:
            h_t,c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]
            # print(h_t)
            # print(c_t)

        for t in range(seq_length):
            x_t = x[:,t,:]
            v_t = torch.cat((h_t,x_t),dim=1)
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))  # output block
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # print(hidden_seq)
        return hidden_seq, (h_t, c_t)

class QLSTMRegression(nn.Module):
    def __init__(self,num_sensors,hidden_units,n_qubits=0,n_qlayers=1):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.lstm = QLSTM(input_size=num_sensors,hidden_size=hidden_units,batch_first=True,n_qubits=n_qubits,n_qlayers = n_qlayers)
        self.linear = nn.Linear(in_features=self.hidden_units,out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out