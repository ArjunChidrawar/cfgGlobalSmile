import os
import stat

parent_dir = '/code/NCLG-MT-master'  # change to your parent directory
stat_info = os.stat(parent_dir)

# Get permission bits, user, group, and others in octal
permissions = stat_info.st_mode
print("Raw mode (in octal):", oct(permissions))

# Check permission bits
readable = bool(permissions & stat.S_IRUSR)
writable = bool(permissions & stat.S_IWUSR)
executable = bool(permissions & stat.S_IXUSR)

print(f"User permissions for {parent_dir}:")
print("  Readable:", readable)
print("  Writable:", writable)
print("  Executable:", executable)