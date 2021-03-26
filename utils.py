import logging
import os


def export_to_shoebox(video_id, input_data, media_folder, ELANType_key, ELANBegin_key, ELANEnd_key):

    fname = f"{video_id}_{ELANType_key}.sht"
    with open(os.path.join(media_folder, fname), "w") as shoeboxfile:

        shoeboxfile.write("\\_sh v3.0  400  ElanExport\n")
        shoeboxfile.write("\\_DateStampHasFourDigitYear\n")
        shoeboxfile.write("\n")
        shoeboxfile.write("\\ELANExport\n")

        for block_id, entry in enumerate(input_data):
            shoeboxfile.write("\n")
            # TODO proper formatting
            shoeboxfile.write(f"\\block {block_id + 1}\n")
            shoeboxfile.write(f"\\{ELANType_key}\n")
            shoeboxfile.write(f"\\ELANBegin {entry[ELANBegin_key]}\n")
            shoeboxfile.write(f"\\ELANEnd {entry[ELANBegin_key]}\n")

    return os.path.join(media_folder, fname)