import os 

# 'origin', 'value_matrix', 'read_content', 'summary', 'pred_prob'
knowledge_growth_list = ['origin', 'value_matrix', 'pred_prob']

# 'sigmoid', 'tanh', 'relu'
add_signal_activation_list = ['sigmoid', 'tanh', 'relu']

# 'sigmoid', 'tanh', 'relu'
erase_signal_activation_list = ['sigmoid', 'relu']

# 'sigmoid', 'tanh', 'relu'
summary_activation_list = ['sigmoid', 'tanh', 'relu']

# 'add_off_erase_off', 'add_off_erase_on', 'add_on_erase_off', 'add_on_erase_on'
write_type_list = ['add_off_erase_on']

for knowledge_growth in knowledge_growth_list:
    for add_signal_activation in add_signal_activation_list:
        for erase_signal_activation in erase_signal_activation_list:
            for write_type in write_type_list:
                for summary_activation in summary_activation_list:
                    args_list = []
                    args_list.append('python main.py --dkvmn_train t --dkvmn_test t --dkvmn_ideal_test t')

                    args_list.append('--knowledge_growth')
                    args_list.append(knowledge_growth)

                    args_list.append('--summary_activation')
                    args_list.append(summary_activation)

                    args_list.append('--add_signal_activation')
                    args_list.append(add_signal_activation)

                    args_list.append('--erase_signal_activation')
                    args_list.append(erase_signal_activation)

                    args_list.append('--write_type')
                    args_list.append(write_type)
                      

                    model = ' '.join(args_list)
                    print(model)
                    os.system(model)
