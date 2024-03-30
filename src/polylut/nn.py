#  This file is part of PolyLUT.
#
#  PolyLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial, reduce

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import itertools
import math, os
import numpy as np
from tqdm import tqdm

from .init import random_restrict_fanin
from .util import fetch_mask_indices, generate_permutation_matrix, generate_permutation_matrix_adder
from .verilog import (
    generate_lut_verilog,
    generate_neuron_connection_verilog,
    generate_neuron_connection_verilog_polylayer,
    generate_neuron_connection_verilog_adder,
    layer_connection_verilog,
    generate_logicnets_verilog,
    generate_register_verilog,
)
from .bench import generate_lut_bench, generate_lut_input_string, sort_to_bench

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    

    
# TODO: Create a container module which performs this function.
# Generate all truth tables for NEQs for a given nn.Module()
def generate_truth_tables(model: nn.Module, verbose: bool = False) -> None:
    training = model.training
    model.eval()
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq_add2:
            if verbose:
                print(colors.RED + f"Calculating truth tables for {name}" + colors.RESET)
            module.calculate_truth_tables_sparselinearneq()
            if verbose:
                print(
                    f"Truth tables generated for {len(module.neuron_truth_tables)} neurons"
                )
                
        elif type(module) == Adder2:
            if verbose:
                print(colors.RED + f"Calculating truth tables for {name}" + colors.RESET)
            module.calculate_truth_tables_adder()
            if verbose:
                print(
                    f"Truth tables generated for {len(module.neuron_truth_tables)} neurons"
                )
           
        elif type(module) == SparseLinearNeq:
            if verbose:
                print(colors.RED + f"Calculating truth tables for {name}" + colors.RESET)
            module.calculate_truth_tables()
            if verbose:
                print(
                    f"Truth tables generated for {len(module.neuron_truth_tables)} neurons"
                )
        
    model.training = training



# TODO: Create a container module which performs this function.
def lut_inference(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq_add2 or type(module) == Adder2  or type(module) == SparseLinearNeq:
            module.lut_inference()
            
            
# TODO: Create a container module which performs this function.
def neq_inference(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if type(module) == SparseLinearNeq:
            module.neq_inference()


# TODO: Should this go in with the other verilog functions?
# TODO: Support non-linear topologies
def module_list_to_verilog_module(
    module_list: nn.ModuleList,
    module_name: str,
    output_directory: str,
    add_registers: bool = True,
    generate_bench: bool = False,
    verbose: bool = False,
):
    input_bitwidth = None
    output_bitwidth = None
    module_contents = ""
    count_layer = 0
    count_adder = 0
    for i in range(len(module_list)):
        m = module_list[i]
        
        if type(m) == SparseLinearNeq:
            if verbose:
                print(f"Generating Verilog for layer_{i}")
            module_prefix = f"layer{i}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(
                module_prefix, output_directory, generate_bench=generate_bench
            )
            if i == 0:
                input_bitwidth = module_input_bits
            elif i == len(module_list) - 1:
                output_bitwidth = module_output_bits
            module_contents += layer_connection_verilog(
                module_prefix,
                input_string=f"M{i}",
                input_bits=module_input_bits,
                output_string=f"M{i+1}",
                output_bits=module_output_bits,
                output_wire=i != len(module_list) - 1,
                register=add_registers,
            )
        elif type(m) == SparseLinearNeq_add2:
            
            if verbose:
                print(f"Generating Verilog for layer_{count_layer}")
            module_prefix = f"layer{count_layer}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(
                module_prefix, output_directory, generate_bench=generate_bench
            )
            if i == 0:
                input_bitwidth = module_input_bits
            elif i == len(module_list) - 1:
                output_bitwidth = module_output_bits
            module_contents += layer_connection_verilog(
                module_prefix,
                input_string=f"M{i}",
                input_bits=module_input_bits,
                output_string=f"M{i+1}",
                output_bits=module_output_bits,
                output_wire=i != len(module_list) - 1,
                register=add_registers,
            )
            count_layer=count_layer+1
        elif type(m) == Adder2:
            if verbose:
                print(f"Generating Verilog for adder_{count_adder}")
            module_prefix = f"adder{count_adder}"
            module_input_bits, module_output_bits = m.gen_layer_verilog(
                module_prefix, output_directory, generate_bench=generate_bench
            )
            if i == 0:
                input_bitwidth = module_input_bits
            elif i == len(module_list) - 1:
                output_bitwidth = module_output_bits
            module_contents += layer_connection_verilog(
                module_prefix,
                input_string=f"M{i}",
                input_bits=module_input_bits,
                output_string=f"M{i+1}",
                output_bits=module_output_bits,
                output_wire=i != len(module_list) - 1,
                register=add_registers,
            )
            count_adder=count_adder+1
        else:
            raise Exception(
                f"Expect type(module) == SparseLinearNeq, {type(module)} found"
            )
    module_list_verilog = generate_logicnets_verilog(
        module_name=module_name,
        input_name="M0",
        input_bits=input_bitwidth,
        output_name=f"M{len(module_list)}",
        output_bits=output_bitwidth,
        module_contents=module_contents,
    )
    reg_verilog = generate_register_verilog()
    with open(f"{output_directory}/myreg.v", "w") as f:
        f.write(reg_verilog)
    with open(f"{output_directory}/{module_name}.v", "w") as f:
        f.write(module_list_verilog)

        
class SparseLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        new_in_features: int,
        degree: int,
        fan_in: int,
        bias: bool = True,
    ) -> None:
        super(SparseLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        y = 1.0 / np.sqrt(new_in_features)
        self.weight = torch.nn.Parameter(torch.randn(out_features, new_in_features))
        self.weight.data.uniform_(-y, y)

    def forward(self, input: Tensor) -> Tensor:
        return (input * self.weight).sum(dim=-1) + self.bias
    
    
class Adder2(nn.Module):
    def __init__(
        self,
        out_features: int,
        ensemble: int,
        input_quant,
        output_quant,
        apply_input_quant=True,
        apply_output_quant=True,
    ) -> None:
        super(Adder2, self).__init__()
        self.ensemble = 2
        self.out_features = out_features
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        self.ensemble = ensemble
        self.is_lut_inference = False
        
    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True):
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.out_features * input_bitwidth * self.ensemble
        M0_bits = input_bitwidth * self.ensemble
        total_output_bits = self.out_features * output_bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        
        print('adder(gen_layer_verilog): input_bitwidth = ' + colors.BLUE + str(input_bitwidth) + colors.RESET)
        print('adder(gen_layer_verilog): output_bitwidth = ' + colors.BLUE + str(output_bitwidth) + colors.RESET)
        print('adder(gen_layer_verilog): total_input_bits = ' + colors.BLUE + str(total_input_bits) + colors.RESET)
        print('adder(gen_layer_verilog): total_output_bits = ' + colors.BLUE + str(total_output_bits) + colors.RESET)
        print('adder(gen_layer_verilog): layer_contents = ' + colors.BLUE + str(layer_contents) + colors.RESET)
        
        
        for index in tqdm(range(self.out_features), desc='gen_neuron_verilog'):
            module_name = f"{module_prefix}_N{index}"
            neuron_verilog = self.gen_neuron_verilog(
                index, module_name
            )  # Generate the contents of the neuron verilog
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(neuron_verilog)
            connection_string = generate_neuron_connection_verilog_adder(
                self.ensemble, input_bitwidth, M0_bits*index
            )  # Generate the string which connects the synapses to this neuron
            wire_name = f"{module_name}_wire"
            connection_line = f"wire [{self.ensemble*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    def gen_neuron_verilog(self, index, module_name):
        (
            #indices,
            input_perm_matrix,
            float_output_states,
            bin_output_states,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = self.ensemble * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        for i in range(num_entries):
            entry_str = ""
            for idx in range(self.ensemble):
                val = input_perm_matrix[i, idx]
                entry_str += self.input_quant.get_bin_str(val)
            res_str = self.output_quant.get_bin_str(bin_output_states[i])
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
        return generate_lut_verilog(
            module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )
    
    
    def lut_inference(self):
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()
    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(
        self,
        connected_input: Tensor,
        input_perm_matrix: Tensor,
        bin_output_states: Tensor,
    ) -> Tensor:
        fan_in_size = self.ensemble
        ci_bcast = connected_input.unsqueeze(2).cuda()  # Reshape to B x Fan-in x 1
        pm_bcast = (
            input_perm_matrix.t().unsqueeze(0).cuda()
        )  # Reshape to 1 x Fan-in x InputStates
        
        eq = (ci_bcast == pm_bcast).sum(
            dim=1
        ) == fan_in_size  # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )
        indices = torch.argmax(eq.type(torch.int64), dim=1)
        out = bin_output_states[indices]

        return out
    
    def lut_forward(self, x_cat: Tensor) -> Tensor:
        if self.apply_input_quant:
            x_cat = self.input_quant(x_cat) 
        x_cat.cuda()
        x_len = x_cat.shape[0]//2
        x1 = x_cat[0:x_len]
        x2 = x_cat[x_len:x_len*2]

        y = torch.zeros((x1.shape[0], self.out_features)).cuda()
        
        # Perform table lookup for each neuron output
        for i in range(self.out_features):
            (
                input_perm_matrix,
                float_output_states,
                bin_output_states,
            ) = self.neuron_truth_tables[i]
            connected_input = torch.cat((x1[:, i].unsqueeze(1), x2[:, i].unsqueeze(1)), dim=1)
            y[:, i] = self.table_lookup(
                connected_input, input_perm_matrix, bin_output_states
            )
        return y
     
    def forward(self, x_cat):
        if self.is_lut_inference:
            output = self.lut_forward(x_cat)
        else:
            x_len = x_cat.shape[0]//2
            x1 = x_cat[0:x_len]
            x2 = x_cat[x_len:x_len*2]

            if self.apply_input_quant:
                x1 = self.input_quant(x1)
                x2 = self.input_quant(x2)

            output = torch.mean(torch.stack((x1, x2), dim=0), dim=0)
            if self.apply_output_quant:
                output = self.output_quant(output)
                
        return output
    
    def forward_to_fill_luts(self, x_cat: Tensor) -> Tensor:
        x_cat = x_cat.to(torch.float32)
        x1 = x_cat[:,0].unsqueeze(1)
        x2 = x_cat[:,1].unsqueeze(1)
        
        if self.apply_input_quant:
            x1 = self.input_quant(x1)
            x2 = self.input_quant(x2)
            
        x1 = x1.repeat(1, self.out_features)
        x2 = x2.repeat(1, self.out_features)
        
        x1 = x1.reshape(x1.shape[0], self.out_features)
        x2 = x2.reshape(x2.shape[0], self.out_features)
        
        output = torch.mean(torch.stack((x1, x2), dim=0), dim=0)
        if self.apply_output_quant:
            output = self.output_quant(output)
        return output
    
    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables_adder(self):
        with torch.no_grad():
            # Precalculate all of the input value permutations
            input_state_space = list()  # TODO: is a list the right data-structure here?
            bin_state_space = list()
            neuron_state_space = (
                self.input_quant.get_state_space()
            )  # TODO: this call should include the index of the element of interest
            bin_space = (
                self.input_quant.get_bin_state_space()
            )  # TODO: this call should include the index of the element of interest

            input_state_space.append(neuron_state_space)
            bin_state_space.append(bin_space)
            neuron_truth_tables = list()

            # Retrieve the possible state space of the current neuron
            connected_state_space = [input_state_space[0] for i in range(self.ensemble)]
            bin_connected_state_space = [bin_state_space[0] for i in range(self.ensemble)]
            # Generate a matrix containing all possible input states
            input_permutation_matrix = generate_permutation_matrix_adder(
                connected_state_space
            ).cuda()  # matrix of all input combinations
            bin_input_permutation_matrix = generate_permutation_matrix_adder(
                bin_connected_state_space
            )

            # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
            apply_input_quant, apply_output_quant = (
                self.apply_input_quant,
                self.apply_output_quant,
            )
            self.apply_input_quant, self.apply_output_quant = False, False
            is_bin_output = self.output_quant.is_bin_output
            self.output_quant.float_output()
            step = input_permutation_matrix.shape[0]
            out_temp = self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            output_states = self.output_quant(out_temp)
            
            
            for segment in range(step, input_permutation_matrix.shape[0], step):
                output_states = torch.cat(
                    (
                        output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.bin_output()
            bin_out_temp = self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            bin_output_states = self.output_quant(bin_out_temp)  # Calculate bin for the current input

            for segment in range(step, input_permutation_matrix.shape[0], step):
                bin_output_states = torch.cat(
                    (
                        bin_output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.is_bin_output = is_bin_output
            self.apply_input_quant, self.apply_output_quant = (
                apply_input_quant,
                apply_output_quant,
            )
            for n in range(self.out_features):
                # Append the connectivity, input permutations and output permutations to the neuron truth tables
                neuron_truth_tables.append(
                    (
                        bin_input_permutation_matrix,
                        output_states[:, n],
                        bin_output_states[:, n],
                    )
                )
        self.neuron_truth_tables = neuron_truth_tables

        
        
class SparseLinearNeq_add2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_quant,
        output_quant,
        mask,
        imask,
        new_in_features,
        fan_in,
        degree,
        apply_input_quant=True,
        apply_output_quant=True,
    ) -> None:
        super(SparseLinearNeq_add2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.mask = mask
        self.imask = imask
        self.new_in_features = new_in_features
        self.fan_in = fan_in
        self.degree = degree
        self.fc1 = SparseLinear(
            in_features, out_features, new_in_features, degree, fan_in
        )
        self.fc2 = SparseLinear(
            in_features, out_features, new_in_features, degree, fan_in
        )
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        
        self.ensemble = int(2)

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    # TODO: Update this code to support custom bitwidths per input/output
    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True):
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features * input_bitwidth
        total_output_bits = self.out_features * output_bitwidth *self.ensemble
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        
        for index in tqdm(range(self.out_features), desc='gen_neuron_verilog'):
            for ensemble_idx in range(self.ensemble):
                module_name = f"{module_prefix}_N{index}_E{ensemble_idx}"
                indices, _, _, _, _, _ = self.neuron_truth_tables[index]
                neuron_verilog = self.gen_neuron_verilog(
                    index, module_name, ensemble_idx
                )  # Generate the contents of the neuron verilog
                with open(f"{directory}/{module_name}.v", "w") as f:
                    f.write(neuron_verilog)
                connection_string = generate_neuron_connection_verilog_polylayer(
                    indices[self.fan_in*ensemble_idx : self.fan_in*(ensemble_idx+1)], 
                    input_bitwidth,
                )  # Generate the string which connects the synapses to this neuron
                wire_name = f"{module_name}_wire"
                connection_line = f"wire [{(len(indices)//self.ensemble)*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
                inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
                layer_contents += connection_line + inst_line
                output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    def gen_neuron_verilog(self, index, module_name, ensemble_idx):
        (
            indices,
            input_perm_matrix,
            float_output_states_1,
            bin_output_states_1,
            float_output_states_2,
            bin_output_states_2,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = (len(indices)//self.ensemble) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        if ensemble_idx == 0:
            for i in range(num_entries):
                entry_str = ""
                for idx in range(len(indices)//self.ensemble):
                    val = input_perm_matrix[i, idx]
                    entry_str += self.input_quant.get_bin_str(val)
                res_str = self.output_quant.get_bin_str(bin_output_states_1[i])
                lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
            return generate_lut_verilog(
                module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
            )
        elif ensemble_idx == 1:
            for i in range(num_entries):
                entry_str = ""
                for idx in range(len(indices)//self.ensemble):
                    val = input_perm_matrix[i, idx]
                    entry_str += self.input_quant.get_bin_str(val)
                res_str = self.output_quant.get_bin_str(bin_output_states_2[i])
                lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
            return generate_lut_verilog(
                module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
            )


    def lut_inference(self):
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()

    def neq_inference(self):
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()

    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(
        self,
        connected_input: Tensor,
        input_perm_matrix: Tensor,
        bin_output_states: Tensor,
    ) -> Tensor:
        fan_in_size = self.fan_in
        ci_bcast = connected_input.unsqueeze(2).cuda()  # Reshape to B x Fan-in x 1
        pm_bcast = (
            input_perm_matrix.t().unsqueeze(0).cuda()
        )  # Reshape to 1 x Fan-in x InputStates
        
        eq = (ci_bcast == pm_bcast).sum(
            dim=1
        ) == fan_in_size  # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )
        indices = torch.argmax(eq.type(torch.int64), dim=1)
        return bin_output_states[indices]

    def lut_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(
                x
            )  # Use this to fetch the bin output of the input, if the input isn't already in binary format
        x.cuda()
        x_de_1 = x.clone()
        x_de_2 = x.clone()
        y1 = torch.zeros((x.shape[0], self.out_features)).cuda()
        y2 = torch.zeros((x.shape[0], self.out_features)).cuda()
        
        # Perform table lookup for each neuron output
        for i in range(self.out_features):
            (
                indices,
                input_perm_matrix,
                float_output_states_1,
                bin_output_states_1,
                float_output_states_2,
                bin_output_states_2,
            ) = self.neuron_truth_tables[i]
            indices.cuda()
            connected_input_1 = x_de_1[:, indices[0:self.fan_in]]
            connected_input_2 = x_de_2[:, indices[self.fan_in:self.fan_in*2]]
            y1[:, i] = self.table_lookup(
                connected_input_1, input_perm_matrix, bin_output_states_1
            )
            y2[:, i] = self.table_lookup(
                connected_input_2, input_perm_matrix, bin_output_states_2
            )
        
        y_cat = torch.cat((y1,y2),0)
        return y_cat

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            x_cat = self.lut_forward(x)
        else:
            if self.apply_input_quant:
                x = self.input_quant(x)
            x_de_1 = x.clone()
            x_de_2 = x.clone()

            x1 = x_de_1[:, self.imask[:,0:self.fan_in]]
            x2 = x_de_2[:, self.imask[:,self.fan_in:self.fan_in*2]]

            x1 = x1.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)
            x2 = x2.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)
            
            x1 = self.fc1(x1)
            x2 = self.fc2(x2)

            if self.apply_output_quant:
                x1 = self.output_quant(x1)
                x2 = self.output_quant(x2)

            x_cat = torch.cat((x1,x2),0)
            

        return x_cat

    def forward_to_fill_luts(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)
        x_de_1 = x.clone()
        x_de_2 = x.clone()
        x_de_1 = x_de_1.repeat(1, self.out_features)
        x_de_2 = x_de_2.repeat(1, self.out_features)
        x_de_1 = x_de_1.reshape(x_de_1.shape[0], self.out_features, self.fan_in)
        x_de_2 = x_de_2.reshape(x_de_2.shape[0], self.out_features, self.fan_in)

        x_de_1 = x_de_1.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)
        x_de_2 = x_de_2.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)

        x_de_1 = self.fc1(x_de_1)
        x_de_2 = self.fc2(x_de_2)

        if self.apply_output_quant:
            x_de_1 = self.output_quant(x_de_1)
            x_de_2 = self.output_quant(x_de_2)
            
        return x_de_1, x_de_2

    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables_sparselinearneq(self):
        with torch.no_grad():
            # Precalculate all of the input value permutations
            input_state_space = list()  # TODO: is a list the right data-structure here?
            bin_state_space = list()
            neuron_state_space = (
                self.input_quant.get_state_space()
            )  # TODO: this call should include the index of the element of interest
            bin_space = (
                self.input_quant.get_bin_state_space()
            )  # TODO: this call should include the index of the element of interest

            input_state_space.append(neuron_state_space)
            bin_state_space.append(bin_space)
            neuron_truth_tables = list()

            # Retrieve the possible state space of the current neuron
            connected_state_space = [input_state_space[0] for i in range(self.fan_in)]
            bin_connected_state_space = [bin_state_space[0] for i in range(self.fan_in)]
            # Generate a matrix containing all possible input states
            input_permutation_matrix = generate_permutation_matrix(
                connected_state_space
            ).cuda()  # matrix of all input combinations
            bin_input_permutation_matrix = generate_permutation_matrix(
                bin_connected_state_space
            )
            # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
            apply_input_quant, apply_output_quant = (
                self.apply_input_quant,
                self.apply_output_quant,
            )
            self.apply_input_quant, self.apply_output_quant = False, False
            is_bin_output = self.output_quant.is_bin_output
            self.output_quant.float_output()
            step = input_permutation_matrix.shape[0]

            
            
            out_temp_1, out_temp_2 = self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            output_states_1 = self.output_quant(out_temp_1)
            output_states_2 = self.output_quant(out_temp_2)

            self.output_quant.bin_output()
            bin_out_temp_1, bin_out_temp_2 = self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            bin_output_states_1 = self.output_quant(bin_out_temp_1)  # Calculate bin for the current input
            bin_output_states_2 = self.output_quant(bin_out_temp_2)  # Calculate bin for the current input
            
            self.output_quant.is_bin_output = is_bin_output
            self.apply_input_quant, self.apply_output_quant = (
                apply_input_quant,
                apply_output_quant,
            )
            for n in range(self.out_features):
                # Append the connectivity, input permutations and output permutations to the neuron truth tables
                neuron_truth_tables.append(
                    (
                        self.imask[n],
                        bin_input_permutation_matrix,
                        output_states_1[:, n],
                        bin_output_states_1[:, n],
                        output_states_2[:, n],
                        bin_output_states_2[:, n],
                    )
                )  # Change this to be the binary output states

        self.neuron_truth_tables = neuron_truth_tables
        
        
        
        
# TODO: Perhaps make this two classes, separating the LUT and NEQ code.
class SparseLinearNeq(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        input_quant,
        output_quant,
        mask,
        imask,
        new_in_features,
        fan_in,
        degree,
        apply_input_quant=True,
        apply_output_quant=True,
    ) -> None:
        super(SparseLinearNeq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.mask = mask
        self.imask = imask
        self.new_in_features = new_in_features
        self.fan_in = fan_in
        self.degree = degree
        self.fc = SparseLinear(
            in_features, out_features, new_in_features, degree, fan_in
        )
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    # TODO: Update this code to support custom bitwidths per input/output
    def gen_layer_verilog(self, module_prefix, directory, generate_bench: bool = True):
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        total_input_bits = self.in_features * input_bitwidth
        total_output_bits = self.out_features * output_bitwidth
        layer_contents = f"module {module_prefix} (input [{total_input_bits-1}:0] M0, output [{total_output_bits-1}:0] M1);\n\n"
        output_offset = 0
        for index in tqdm(range(self.out_features), desc='gen_neuron_verilog'):
            module_name = f"{module_prefix}_N{index}"
            indices, _, _, _ = self.neuron_truth_tables[index]
            neuron_verilog = self.gen_neuron_verilog(
                index, module_name
            )  # Generate the contents of the neuron verilog
            with open(f"{directory}/{module_name}.v", "w") as f:
                f.write(neuron_verilog)
            if generate_bench:
                neuron_bench = self.gen_neuron_bench(
                    index, module_name
                )  # Generate the contents of the neuron verilog
                with open(f"{directory}/{module_name}.bench", "w") as f:
                    f.write(neuron_bench)
            connection_string = generate_neuron_connection_verilog(
                indices, input_bitwidth
            )  # Generate the string which connects the synapses to this neuron
            wire_name = f"{module_name}_wire"
            connection_line = f"wire [{len(indices)*input_bitwidth-1}:0] {wire_name} = {{{connection_string}}};\n"
            inst_line = f"{module_name} {module_name}_inst (.M0({wire_name}), .M1(M1[{output_offset+output_bitwidth-1}:{output_offset}]));\n\n"
            layer_contents += connection_line + inst_line
            output_offset += output_bitwidth
        layer_contents += "endmodule"
        with open(f"{directory}/{module_prefix}.v", "w") as f:
            f.write(layer_contents)
        return total_input_bits, total_output_bits

    # TODO: Move the verilog string templates to elsewhere
    # TODO: Move this to another class
    def gen_neuron_verilog(self, index, module_name):
        (
            indices,
            input_perm_matrix,
            float_output_states,
            bin_output_states,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        for i in range(num_entries):
            entry_str = ""
            for idx in range(len(indices)):
                val = input_perm_matrix[i, idx]
                entry_str += self.input_quant.get_bin_str(val)
            res_str = self.output_quant.get_bin_str(bin_output_states[i])
            lut_string += f"\t\t\t{int(cat_input_bitwidth)}'b{entry_str}: M1r = {int(output_bitwidth)}'b{res_str};\n"
        return generate_lut_verilog(
            module_name, int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )

    # TODO: Move the string templates to bench.py
    # TODO: Move this to another class
    def gen_neuron_bench(self, index, module_name):
        (
            indices,
            input_perm_matrix,
            float_output_states,
            bin_output_states,
        ) = self.neuron_truth_tables[index]
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        cat_input_bitwidth = len(indices) * input_bitwidth
        lut_string = ""
        num_entries = input_perm_matrix.shape[0]
        # Sort the input_perm_matrix to match the bench format
        input_state_space_bin_str = list(
            map(
                lambda y: list(map(lambda z: self.input_quant.get_bin_str(z), y)),
                input_perm_matrix,
            )
        )
        sorted_bin_output_states = sort_to_bench(
            input_state_space_bin_str, bin_output_states
        )
        # Generate the LUT for each output
        for i in range(int(output_bitwidth)):
            lut_string += f"M1[{i}]       = LUT 0x"
            output_bin_str = reduce(
                lambda b, c: b + c,
                map(
                    lambda a: self.output_quant.get_bin_str(a)[
                        int(output_bitwidth) - 1 - i
                    ],
                    sorted_bin_output_states,
                ),
            )
            lut_hex_string = f"{int(output_bin_str,2):0{int(num_entries/4)}x} "
            lut_string += lut_hex_string
            lut_string += generate_lut_input_string(int(cat_input_bitwidth))
        return generate_lut_bench(
            int(cat_input_bitwidth), int(output_bitwidth), lut_string
        )

    def lut_inference(self):
        self.is_lut_inference = True
        self.input_quant.bin_output()
        self.output_quant.bin_output()

    def neq_inference(self):
        self.is_lut_inference = False
        self.input_quant.float_output()
        self.output_quant.float_output()

    # TODO: This function might be a useful utility outside of this class..
    def table_lookup(
        self,
        connected_input: Tensor,
        input_perm_matrix: Tensor,
        bin_output_states: Tensor,
    ) -> Tensor:
        fan_in_size = self.fan_in
        ci_bcast = connected_input.unsqueeze(2).cuda()  # Reshape to B x Fan-in x 1
        pm_bcast = (
            input_perm_matrix.t().unsqueeze(0).cuda()
        )  # Reshape to 1 x Fan-in x InputStates
        eq = (ci_bcast == pm_bcast).sum(
            dim=1
        ) == fan_in_size  # Create a boolean matrix which matches input vectors to possible input states
        matches = eq.sum(dim=1)  # Count the number of perfect matches per input vector
        if not (matches == torch.ones_like(matches, dtype=matches.dtype)).all():
            raise Exception(
                f"One or more vectors in the input is not in the possible input state space"
            )
        indices = torch.argmax(eq.type(torch.int64), dim=1)
        return bin_output_states[indices]

    def lut_forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(
                x
            )  # Use this to fetch the bin output of the input, if the input isn't already in binary format
        x.cuda()
        y = torch.zeros((x.shape[0], self.out_features)).cuda()
        # Perform table lookup for each neuron output
        for i in range(self.out_features):
            (
                indices,
                input_perm_matrix,
                float_output_states,
                bin_output_states,
            ) = self.neuron_truth_tables[i]
            indices.cuda()
            connected_input = x[:, indices]
            step = 64
            y[:, i] = self.table_lookup(
                connected_input, input_perm_matrix, bin_output_states
            )
        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.is_lut_inference:
            x = self.lut_forward(x)
        else:
            if self.apply_input_quant:
                x = self.input_quant(x)
            x = x[:, self.imask]
            x = x.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)
            x = self.fc(x)
            if self.apply_output_quant:
                x = self.output_quant(x)
        return x

    def forward_to_fill_luts(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)
        x = x.repeat(1, self.out_features)
        x = x.reshape(x.shape[0], self.out_features, self.fan_in)
        x = x.unsqueeze(dim=-2).pow(self.mask).prod(dim=-1)
        x = self.fc(x)
        if self.apply_output_quant:
            x = self.output_quant(x)
        return x

    # Consider using masked_select instead of fetching the indices
    def calculate_truth_tables(self):
        with torch.no_grad():
            # Precalculate all of the input value permutations
            input_state_space = list()  # TODO: is a list the right data-structure here?
            bin_state_space = list()
            neuron_state_space = (
                self.input_quant.get_state_space()
            )  # TODO: this call should include the index of the element of interest
            bin_space = (
                self.input_quant.get_bin_state_space()
            )  # TODO: this call should include the index of the element of interest
            input_state_space.append(neuron_state_space)
            bin_state_space.append(bin_space)
            neuron_truth_tables = list()

            # Retrieve the possible state space of the current neuron
            connected_state_space = [input_state_space[0] for i in range(self.fan_in)]
            bin_connected_state_space = [bin_state_space[0] for i in range(self.fan_in)]
            # Generate a matrix containing all possible input states
            input_permutation_matrix = generate_permutation_matrix(
                connected_state_space
            ).cuda()  # matrix of all input combinations
            bin_input_permutation_matrix = generate_permutation_matrix(
                bin_connected_state_space
            )

            # TODO: Update this block to just run inference on the fc layer, once BN has been moved to output_quant
            apply_input_quant, apply_output_quant = (
                self.apply_input_quant,
                self.apply_output_quant,
            )
            self.apply_input_quant, self.apply_output_quant = False, False
            is_bin_output = self.output_quant.is_bin_output
            self.output_quant.float_output()
            step = input_permutation_matrix.shape[0]
            output_states = self.output_quant(
                self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            )
            for segment in range(step, input_permutation_matrix.shape[0], step):
                output_states = torch.cat(
                    (
                        output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.bin_output()
            bin_output_states = self.output_quant(
                self.forward_to_fill_luts(input_permutation_matrix[0:step, :])
            )  # Calculate bin for the current input
            for segment in range(step, input_permutation_matrix.shape[0], step):
                bin_output_states = torch.cat(
                    (
                        bin_output_states,
                        self.output_quant(
                            self.forward_to_fill_luts(
                                input_permutation_matrix[segment : segment + step, :]
                            )
                        ),
                    ),
                    0,
                )  # Calculate float for the current input
            self.output_quant.is_bin_output = is_bin_output
            self.apply_input_quant, self.apply_output_quant = (
                apply_input_quant,
                apply_output_quant,
            )
            for n in range(self.out_features):
                # Append the connectivity, input permutations and output permutations to the neuron truth tables
                neuron_truth_tables.append(
                    (
                        self.imask[n],
                        bin_input_permutation_matrix,
                        output_states[:, n],
                        bin_output_states[:, n],
                    )
                )  # Change this to be the binary output states
        self.neuron_truth_tables = neuron_truth_tables


def InputTerms(fan_in, degree):
    return list(
        itertools.chain(
            *[
                list(itertools.combinations_with_replacement(range(fan_in), d + 1))
                for d in range(degree)
            ]
        )
    )


def PolyMask(fan_in, degree, terms):
    T = len(terms)
    mask = torch.zeros((T, fan_in)).cuda()
    for t, ks in enumerate(terms):
        for k in ks:
            mask[t, k] += 1
    return mask


def FeatureMask(in_features: int, out_features: int, fan_in: int, degree: int):
    imask = torch.zeros((out_features, fan_in), dtype=torch.long).cuda()
    for i in range(out_features):
        imask[i, :] = torch.randperm(in_features)[:fan_in]
        imask = torch.sort(imask, 1).values
    return imask



class DenseMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 1.0)

    def forward(self):
        return self.mask


class RandomFixedSparsityMask2D(nn.Module):
    def __init__(self, in_features: int, out_features: int, fan_in: int) -> None:
        super(RandomFixedSparsityMask2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = fan_in
        self.mask = Parameter(
            torch.Tensor(out_features, in_features), requires_grad=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.mask, 0.0)
        for i in range(self.out_features):
            x = torch.randperm(self.in_features)[: self.fan_in]
            self.mask[i][x] = 1

    def forward(self):
        return self.mask


class ScalarScaleBias(nn.Module):
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("weight", None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("bias", None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    # Change the default initialisation for BatchNorm
    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class ScalarBiasScale(ScalarScaleBias):
    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias
        if self.weight is not None:
            x = x * self.weight
        return x
