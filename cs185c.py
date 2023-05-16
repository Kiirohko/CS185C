# ----------------------------------------------------------------------
# Name:       CS 185C / BIOL 145 - Final Project
# Purpose:    global pairwise alignment and reporting
# Author(s):  Kevin Huang
# Date:       05/13/23
# ----------------------------------------------------------------------
"""
Global pairwise alignment for nucleotide sequences.

Prompt the user for a fasta file (containing nucleotide sequences).
Expected defline format is >accession_scientific_name common_name order.
Prompt the user to choose a sequence to use as the query.
Perform global pairwise alignment of chosen query sequence against
each other sequence (subjects).
Generate 4 report files from the compiled alignment results.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_fasta(fasta):
    """
    Read a fasta file, extracting the accession numbers and
    corresponding sequences and additional information.
    :param fasta: (string) fasta file name
    :return: tuple containing dictionary of format
            accession: sequence and additional information,
            and list of accession numbers
    """
    # read each line of .fasta file into list
    with open(fasta, 'r', encoding='utf-8') as fasta_file:
        lines = [line.strip() for line in fasta_file.readlines()]

    fasta_dict = {}
    # process first defline of .fasta file
    current_accession, current_information = process_defline(lines[0])
    current_sequence = ''

    # process each line to compose dictionary of accession information
    for line in lines[1:]:
        # reading defline
        if line.startswith('>'):
            # add previous entry to dictionary
            fasta_dict[current_accession] = current_sequence, \
                len(current_sequence), current_information
            # process current defline
            current_accession, current_information = process_defline(line)
            current_sequence = ''
        # reading sequence
        else:
            current_sequence += line

    # reached end of .fasta file, add final entry to dictionary
    fasta_dict[current_accession] = current_sequence, \
        len(current_sequence), current_information

    # compose list of accession numbers
    accession_list = list(fasta_dict)

    return fasta_dict, accession_list


def process_defline(defline):
    """
    Process and extract information from the specially formatted
    fasta deflines.
    :param defline: (string) the defline to process
    :return: tuple containing the accession number (string) and
            tuple of scientific name, common name, and taxonomic order
    """
    # expected defline format:
    # >accession_scientific_name common_name order
    defline_list = defline.split()

    full_accession = defline_list[0].strip('>').split('_')
    accession = full_accession[0]
    scientific_name = ' '.join(full_accession[1:])

    common_name = defline_list[1].replace('_', ' ')
    order = defline_list[2]

    return accession, (scientific_name, common_name, order)


def score_pair(a, b):
    """
    Score the alignment between a pair of nucleotides: +1 for a match,
    -1 for a mismatch, and -2 for a gap (indel).
    :param a: (string) the first nucleotide
    :param b: (string) the second nucleotide
    :return: (integer) the alignment score
    """
    # match -> +1 score
    if a == b:
        return 1
    else:
        # mismatch -> -1 score, gap (indel) -> -2 score
        return -2 if '-' in {a, b} else -1


def global_align(seq_a, seq_b):
    """
    Perform a global pairwise alignment between two sequences using the
    Needleman-Wunsch algorithm, returning the aligned sequences and the
    overall alignment score.
    :param seq_a: (string) the first sequence to align
    :param seq_b: (string) the second sequence to align
    :return: tuple of aligned sequences (string)
            and alignment score (integer)
    """
    gap_penalty = -2
    m = len(seq_a)
    n = len(seq_b)

    # initialize empty grid to build similarity matrix
    grid = np.zeros((m + 1, n + 1))
    # add penalty for lengthening gap in sequence A
    for i in range(m + 1):
        grid[i][0] = gap_penalty * i
    # add penalty for lengthening gap in sequence B
    for j in range(n + 1):
        grid[0][j] = gap_penalty * j
    # fill in rest of cells
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = grid[i - 1][j - 1] + score_pair(seq_a[i - 1], seq_b[j - 1])
            delete = grid[i - 1][j] + gap_penalty
            insert = grid[i][j - 1] + gap_penalty
            # fill cell with optimal score
            grid[i][j] = max(match, delete, insert)

    align_a = ''
    align_b = ''
    i = m
    j = n
    align_score = 0

    # begin traceback to build aligned sequences
    while i > 0 or j > 0:
        # gap in sequence B
        if i > 0 and grid[i][j] == grid[i - 1][j] + gap_penalty:
            align_a = seq_a[i - 1] + align_a
            align_b = '-' + align_b
            align_score += gap_penalty
            i -= 1
        # gap in sequence A
        elif j > 0 and grid[i][j] == grid[i][j - 1] + gap_penalty:
            align_a = '-' + align_a
            align_b = seq_b[j - 1] + align_b
            align_score += gap_penalty
            j -= 1
        # no gap in either sequence
        else:
            align_a = seq_a[i - 1] + align_a
            align_b = seq_b[j - 1] + align_b
            align_score += score_pair(seq_a[i - 1], seq_b[j - 1])
            i -= 1
            j -= 1

    return align_a, align_b, align_score


def alignment_report(align_a, align_b):
    """
    Report on various statistics given a pair of aligned sequences.
    :param align_a: (string) the first aligned sequence
    :param align_b: (string) the second aligned sequence
    :return: alignment length (integer), number of matches (integer),
            % identity (float), and dictionary of format
            region index (start, stop): region sequence
    """
    align_length = len(align_a)
    mismatch_count = 0
    diff_dict = {}
    region_a = ''
    region_b = ''
    region_start = 0
    recording_diff = False

    # traverse aligned sequences to build report on differences
    for p in range(align_length):
        # start of new mismatch region
        if align_a[p] != align_b[p] and not recording_diff:
            recording_diff = True
            mismatch_count += 1
            region_start = p
            region_a += align_a[p]
            region_b += align_b[p]
        # currently inside mismatch region
        elif align_a[p] != align_b[p] and recording_diff:
            mismatch_count += 1
            region_a += align_a[p]
            region_b += align_b[p]
        else:
            # end of current mismatch region
            if recording_diff:
                recording_diff = False
                region_end = p - 1
                diff_dict[(region_start, region_end)] = \
                    (region_a, region_b)
                region_a = ''
                region_b = ''

    match_count = align_length - mismatch_count
    identity = round(match_count / align_length * 100, 2)

    return align_length, match_count, identity, diff_dict


def main():
    # prompt user for .fasta file
    fasta = ''
    while not (fasta.endswith('.fasta') and os.path.exists(fasta)):
        fasta = input('Please specify a .fasta filename: ')

    fasta_dict, accession_list = read_fasta(fasta)

    # display query options to user
    accession_count = len(accession_list)
    print('Query sequence options:')
    for i in range(1, accession_count + 1):
        information = fasta_dict[accession_list[i - 1]][2]
        scientific_name, common_name, order = information
        print(f'{i} - {scientific_name} ({common_name}, {order})')

    # prompt user for query option
    query_accession = ''
    while not query_accession:
        index = input(f'Please select a query option '
                      f'(1 to {accession_count}): ')
        try:
            query_index = int(index)
        except ValueError:
            print(f'Error: invalid argument value')
        else:
            if 1 <= query_index <= accession_count:
                # remove query from accession list, rest are subjects
                query_accession = accession_list.pop(query_index - 1)

    query_sequence, query_length, query_information = \
        fasta_dict[query_accession]

    query_scientific_name, query_common_name, query_order = \
        query_information

    prefix = query_scientific_name.replace(' ', '_')
    txt = f'{prefix}.txt'
    csv = f'{prefix}.csv'

    print(f'Generating alignment report for {query_scientific_name} '
          f'({query_common_name})...')

    # write alignment report into .txt file
    # write organism information into .csv file
    with open(txt, 'w', encoding='utf-8') as sys.stdout:
        with open(csv, 'w', encoding='utf-8') as csv_file:
            # write column headers of .csv file
            csv_file.write('Accession Number,Scientific Name,Common Name,'
                           'Order,Original Length Difference,Identity,'
                           'Mismatch Region Count\n')

            # write header of .txt file
            print(f'{query_accession} - {query_scientific_name} '
                  f'({query_common_name}) - Alignment Results:\n')

            # compose report for query aligned against subjects
            for accession in accession_list:
                subject_sequence, subject_length, subject_information = \
                    fasta_dict[accession]
                original_length_diff = abs(query_length - subject_length)

                subject_scientific_name, subject_common_name, \
                    subject_order = subject_information

                alignment = global_align(query_sequence, subject_sequence)

                print(f'Query aligned against {accession} -'
                      f' {subject_scientific_name} ({subject_common_name})')
                print(f'Query:   {alignment[0]}')
                print(f'Subject: {alignment[1]}')
                print(f'\tAlignment Score: {alignment[2]}')

                align_diff = alignment_report(alignment[0], alignment[1])
                align_length, match_count, identity, diff_dict = align_diff

                print('Alignment Report:')
                print(f'\tAligned Sequence Length: {align_length}')
                print(f'\tOriginal Length Difference: {original_length_diff}')
                print(f'\tMatch Count: {match_count}')
                print(f'\t% Identity: {identity}%')
                if diff_dict:
                    print('Mismatch Regions (Index: Mutation)')
                    for region in diff_dict:
                        if region[0] == region[1]:
                            print(f'\t{region[0]}: '
                                  f'{diff_dict[region][0]} '
                                  f'<-> {diff_dict[region][1]}')
                        else:
                            print(f'\t{region[0]}-{region[1]}: '
                                  f'{diff_dict[region][0]} '
                                  f'<-> {diff_dict[region][1]}')
                mismatch_region_count = len(diff_dict)
                print(f'Mismatch Region Count: {mismatch_region_count}\n')

                csv_file.write(f'{accession},{subject_scientific_name},'
                               f'{subject_common_name},{subject_order},'
                               f'{original_length_diff},{identity}'
                               f',{mismatch_region_count}\n')

    # read .csv file into pandas dataframe
    coral_df = pd.read_csv(csv)

    # map organism orders to colors for generating comparative bar plots
    order_color = {'Actiniaria': 'yellow', 'Corallimorpharia': 'red',
                   'Scleractinia': 'orange', 'Alcyonacea': 'cyan',
                   'Helioporacea': 'blue', 'Pennatulacea': 'purple',
                   'OUTGROUP': 'black'}

    # plot % identity for each subject
    identity_df = coral_df.sort_values(by='Identity')
    entry_order = identity_df['Order'].to_list()
    colors = [order_color[e] for e in entry_order]
    ax = identity_df.plot.barh(title=f'{query_scientific_name} '
                                     f'- Identity Comparison',
                               x='Scientific Name', y='Identity',
                               xlabel='% Identity', ylabel='Subject',
                               color=colors, legend=False)
    ax.bar_label(ax.containers[0])
    plt.tight_layout()
    plt.savefig(f'{prefix}_identity.png', dpi=300)

    # plot mismatch region count for each subject
    mismatch_df = coral_df.sort_values(
        by='Mismatch Region Count', ascending=False)
    entry_order = mismatch_df['Order'].to_list()
    colors = [order_color[e] for e in entry_order]
    ax = mismatch_df.plot.barh(title=f'{query_scientific_name} '
                                     f'- Mismatch Comparison',
                               x='Scientific Name', y='Mismatch Region Count',
                               xlabel='Mismatch Region Count',
                               ylabel='Subject',
                               color=colors, legend=False)
    ax.bar_label(ax.containers[0])
    plt.tight_layout()
    plt.savefig(f'{prefix}_mismatch.png', dpi=300)


if __name__ == '__main__':
    main()
