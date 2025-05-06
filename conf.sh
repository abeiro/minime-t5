#!/bin/bash
clear
cat << EOF
MINIME-T5

This will install MiniMe-T5. This is a tiny LLM (400MB) that will help other "dumb" LLM's perform actions. 

Remember to enable MINIME_T5 in the configuration wizard under the default profile!

Can either run it off your GPU or CPU. CPU is slower but saves precious GPU space. 

If you are not sure use CPU.

EOF

if [ ! -d /home/dwemer/python-minime ]; then
	exit "MiniMe-t5 not installed"
fi

mapfile -t files < <(find /home/dwemer/minime-t5/ -name "start-*.sh")
# Check if any files were found

if [ ${#files[@]} -eq 0 ]; then
    echo "No files found matching the pattern."
    exit 1
fi

# Display the files in a numbered list
echo -e "Select a an option from the list:\n\n"
for i in "${!files[@]}"; do
    echo "$((i+1)). ${files[$i]}"
done

echo "0. Disable Service";
echo

# Prompt the user to make a selection
read -p "Select an option by picking the matching number: " selection

# Validate the input

if [ "$selection" -eq "0" ]; then
    echo "Disabling service. Run this script again to enable"
    rm /home/dwemer/minime-t5/start.sh &>/dev/null
    exit 0
fi

if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#files[@]} ]; then
    echo "Invalid selection."
    exit 1
fi

# Get the selected file
selected_file="${files[$((selection-1))]}"

echo "You selected: $selected_file"

ln -sf $selected_file /home/dwemer/minime-t5/start.sh




