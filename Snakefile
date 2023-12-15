rule all:
    input:
        "results/final_result.png"



rule extract_paper_information:
    input:
        "paper_information.txt"
    output:
        "results/dimension.txt",
        "results/initial_condition.txt",
        "results/time_range.txt",
        "results/boundary_condition.txt",
        "results/properties.txt"
    conda:
        "envs/fenicsx-env.yaml"
    shell:
        "python3 quenching.py {input} {output}"

rule paper_result:
    input:
        "paper_information.txt"
    output:
        "paper_result.png"  # this is screenshot from the journal paper directly
    shell:
        ""        
        
rule generate_my_result:
    input:
        "results/dimension.txt",
        "results/initial_condition.txt",
        "results/time_range.txt",
        "results/boundary_condition.txt",
        "results/properties.txt"
    output:
        "results/my_result.png"
    conda:
        "envs/fenicsx-env.yaml"
    shell:
        "python3 quenching.py {input} {output}"
        
rule compare_result:
    input:
        "paper_result.png",
        "results/my_result.png"
    output:
        "results/final_result.png"
    conda:
        "envs/compare.yaml"
    shell:
        "python3 compare.py {input} {output}"


