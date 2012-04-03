def load_arrayexpress():
    if(os.path.exists('array_express.data')):
        print "Loading cached array express data..."
        r = util.load_rep('array_express.data')
    else:
        print "Loading array express data..."
        aexpress = wrapper_tsv.read_tsv('/home/marc/Databases/listdl.tab')
        
        print "Grouping array express data on gene identifier and condition..."
        aexpress = aexpress.group_by(_.gene_identifier, _.factor_value, keep={1:("experimental_factor",)}).get(_.gene_identifier, _.factor_value, _.experimental_factor, _.expression).copy()
        ae_down = numpy.array(map(list, aexpress.expression[_ == "DOWN"].count()()),dtype=int)
        ae_up = numpy.array(map(list, aexpress.expression[_ == "UP"].count()()),dtype=int)
        ae_count = ae_up - ae_down
        ae_genenames = dict([(name, gpos) for gpos, name in enumerate(aexpress.gene_identifier())])
        
        #normalize per condition
        ae_count = ae_count / ae_count.sum(axis=0)

        eset = aexpress.experimental_factor.set()()
        ecounts = []
        ename = []
        for name in eset:
            eindex = aexpress.experimental_factor.get(pos()/"pos",_)[_.experimental_factor == name].pos()
            ecounts.append(ae_count[:,eindex])
            ename.append(name)
         
        r = (ecounts, ae_genenames, ename)
        util.save_rep(r, 'array_express.data')
    return r


def load_chipchip(gn_to_orf):
    if(os.path.exists('chipchip.data')):
        print "Loading cached chipchip data..."
        res = util.load_rep('chipchip.data')
    else:
        print "Parsing and combining chipchip data..."
        chipchip = wrapper_chipchip.read_chipchip('/home/marc/Databases/chip-chip/');

        print "Loading chipchip data and matching it to gene names..."
        #get orf code for chipchip regulator name by matching with gn list. Filter out non matched regulators, return first match (should all be equal)
        #chipchip_orf = gn_to_orf[{_.gene_name:chipchip.regulator}].hstack(chipchip)[_.gene_name]
        chipchip_orf = chipchip.match(gn_to_orf,"regulator","gene_name", group=True)[_.orf_code.count() > 0].get(_.orf_code[0], _.gene_name[0],  "~")

        #flatten chipchip list, rename orf_code field to orf_reg
        cc = chipchip_orf.flat()/{"orf_code":"orf_reg"}

        #filter out orfs not available in sgd orfs list
        cc = cc.match(gn_to_orf, "orfs", "gene_name", group=True)[_.orf_code.count() > 0].get(_.orf_code[0]/"orfs", "~").copy()
        
        yt = load_yeastract(gn_to_orf)
        cc = (cc.get(_.orf_reg, _.orfs, _.max_pval, _.conservation).tuple().array() + yt.join(rep([(-1.0, -1.0)])).tuple().array()).E.A.copy()

        ccdata = cc.get(_.max_pval, _.conservation).tuple()()
        ccnames = cc.get(_.orf_reg, _.orfs)()
        res = (ccdata, ccnames)
        util.save_rep(res, 'chipchip.data')
    return res


def load_phospho(gn_to_orf):
    pp = wrapper_phospho.read_phospho('/home/marc/Databases/phospho/')
    s = numpy.std(pp.score().view(numpy.ndarray))
    pp = pp.match(gn_to_orf, "kinase", "gene_name").get(_.orf_code/"kinase_orf", _.orf, (_.score/s)/"score")
    pp = pp.match(gn_to_orf, "orf", "gene_name").get(_.kinase_orf, _.orf_code/"target_orf", _.score).copy()
    return pp

def load_yeastract(gn_to_orf):
    yt = wrapper_tsv.read_tsv('/home/marc/Databases/yeastract/regul.tsv')/("tf_name", "regul_name")
    yt = yt.map(lambda x:x.upper(),otype="string[]")
    yt = yt.match(gn_to_orf, "tf_name", "gene_name").get(_.orf_code/"tf_orf", _.regul_name)
    yt = yt.match(gn_to_orf, "regul_name", "gene_name").get(_.tf_orf, _.orf_code/"regul_orf").copy()
    return yt

def add_protprot_edges(network, gn_to_orf):
    if(os.path.exists('protprot.data')):
        res = util.load_rep('protprot.data')
    else:
        print "Loading yeast protein-protein data..."
        yeast_bc = wrapper_psimi.read_psimi('yeast.xml').copy()

        print "Mapping interactor names to  gene orf names..."
        interactors = yeast_bc.interactors.A.match(gn_to_orf, "short_name", "gene_name").get(_.id, _.orf_code).copy()

        print "Unpacking interactions and filtering genetic interference interactions..."
        interactions = yeast_bc.interactions.get(_.biogrid_type, _.participants.get(_.interactor_id, _.role), _.experiments.flat()).copy()
        experiments = yeast_bc.experiments[_.method_name != "genetic interference"].id
        interactions = interactions.match(experiments, "experiments", "id").L.copy()

        print "Unpacking baits and preys..."
        interactions = interactions.get(_.biogrid_type, _[_.role == "bait"].interactor_id[0]/"bait", _[_.role == "prey"].interactor_id/"prey").flat("prey").copy()
        
        print "Matching interactions with interactors..."
        interactions = interactions.match(interactors/{"orf_code": "bait_orf"}, "bait", "id").match(interactors/{"orf_code":"prey_orf"}, "prey", "id").get(_.biogrid_type, _.bait_orf, _.prey_orf).copy()

        print "Grouping data on interaction pair and measurement type and count different measurement types per interaction pair"
        interactions = interactions.join(rep([1.0])/"score").copy()
        
        pp = load_phospho(gn_to_orf)
        interactions = (interactions.get("bait_orf", "prey_orf", "biogrid_type", "score").tuple().array() + interactions.get("prey_orf", "bait_orf", "biogrid_type", "score").tuple().array())
        interactions = interactions + pp.join(rep(["kinase"])/"biogrid_type").get(_.kinase_orf, _.target_orf, _.biogrid_type, _.score).tuple().array() 

        res = interactions.E.A.get(_.get(_.bait_orf, _.prey_orf).tuple()/"pair", _.biogrid_type, _.score).group_by(_.pair, _.biogrid_type)
        res = res.get(_.pair.A, _.biogrid_type, _.score.sum()).copy()

        icount = res.group_by(_.bait_orf).get(_.bait_orf, _.prey_orf.set().E.count()/"interaction_count")
        res = res.match(icount, "bait_orf", "bait_orf")/{"interaction_count":"bait_count"}
        res = res.match(icount/{"bait_orf":"prey_orf"}, "prey_orf", "prey_orf")/{"interaction_count":"prey_count"}
        res = res.get(_.bait_orf, _.prey_orf, _.biogrid_type, _.score, _.bait_count, _.prey_count).copy()()
        #network.addEdges("protprot", ccnames[1], ccnames[0], res, cctypes)
        res[4].shape = res[4].shape + (1,)
        res[5].shape = res[5].shape + (1,)

        dres = numpy.concatenate((numpy.array(map(list,res[3]), dtype="uint16"), res[4], res[5]), axis=1)        
        #filter = ((dres[:,-1] < 10) & (dres[:,-2] < 10)) | (dres[:,-3] != 0)
        res = (res[0], res[1], res[2], dres[:,:-2])
        #res = (res[0][filter], res[1][filter], res[2], dres[filter,:-2])

        util.save_rep(res, 'protprot.data')
    network.addEdges("protprot", res[0], res[1], res[3], list(res[2]), [0] * len(res[2]))
#    network.addEdges("protprot", nres[0], nres[1], nres[3], list(nres[2]) + ['bait_count', 'prey_count'])
    return res
    

def load_gntoorf():
    print "Loading gene orf table..."
    gn = wrapper_tsv.read_tsv('/home/marc/Databases/SGD_features.tab', 
            fieldnames=('sgdid', 'feat_type', 'feat_qual', 'feat_name', 'gene_name', 'alias', 
                        'parent_feat_name', 'sgdid2', 'chromosome', 'start', 'stop', 'strand' ,
                        'genetic_pos', 'coordinate_version', 'sequence_version', 'description'));    
    
    gn = gn[_.feat_type == "ORF"]/{'feat_name':'orf_code', 'alias':'alt_gene_names'}
    #Split alt_gene names
    print "Loading gene names and creating name translation table..."
    gn = gn.get(_.gene_name, _.alt_gene_names.map(lambda x: str.split(x,'|'), otype="[symonyms:~](bytes[])").E, "~").copy()

    #create a list from gene names, alt genenames and orf code to of code
    gn_to_orf = (gn.get(_.gene_name, _.orf_code).tuple().array() + 
                 gn.get(_.alt_gene_names, _.orf_code).flat().tuple().array() + 
                 gn.get(_.orf_code, _.orf_code).tuple().array()).E.A.copy()
    
    gn_to_orf = gn_to_orf[_.gene_name != ""].group_by(_.gene_name).get(_.gene_name, _.orf_code[0]).copy()

    return (gn, gn_to_orf)

def addprotein_dna_edges(network, gn_to_orf):
    if(os.path.exists('protdna.data')):
        print "Loading cached protein dna edges..."
        res, ccnames, cctypes = util.load_rep('protdna.data')
    else:
        ae_counts, ae_genenames, ae_expfactor = load_arrayexpress()
        ccdata, ccnames = load_chipchip(gn_to_orf)

        #add chipchip data, based on out orfs, in orfs, and feature data (pval, conservation)
        print "Adding protein-dna edges..."
        set_ccdata = util.darray(list(set(ccdata)),object,1)
        cctypes = []
        res = numpy.zeros((len(ccdata), len(set_ccdata) + len(ae_expfactor)),dtype=float)
        for tpos in xrange(len(set_ccdata)):
            cctype = set_ccdata[tpos:tpos + 1]
            res[numpy.equal(ccdata,cctype),len(cctypes)] = 1
            cctypes.append("({0:.3f}, {1})".format(cctype[0][0],int(cctype[0][1])))

        arraypos = len(set_ccdata)

        for cpos, (regname, targetname) in enumerate(zip(*ccnames)):
            if(regname in ae_genenames and targetname in ae_genenames):
                posl = ae_genenames[regname]
                posr = ae_genenames[targetname]
                for colpos, ae_count in enumerate(ae_counts):
                    corr = numpy.corrcoef(ae_count[posl,:], ae_count[posr,:])[0,1]
                    if(not numpy.isnan(corr)):
                        res[cpos,arraypos + colpos] = numpy.abs(corr) ** 2
        cctypes.extend(ae_expfactor)
        xsave = (res, ccnames, cctypes)
        util.save_rep(xsave, 'protdna.data')
    network.addEdges("protdna", ccnames[0], ccnames[1], res, cctypes, [0] * len(cctypes))


def load_network():
    #reading in the data
    print "Loading knockout experiment table..."
    experiment_list, hughes_data, control_data = wrapper_hughes.read_hughes('/home/marc/Databases/hughes');

    gn, gn_to_orf = load_gntoorf()
    print "Loading hughes control data and mapping it to gene names and removing double orfs..."
    #create measurment nodes: determine which orfs were measured in the hughes dataset
    cgenes = control_data.match(gn, "gene_official", "orf_code")
    #remove double occuring orf
    cgenes = cgenes.group_by(_.orf_code)[_.gene_official.count() == 1].flat("gene_official").copy()

    print "Creating network nodes..."
    #create nodes
    node_ids = gn.get(_.orf_code/"nodeid", _.gene_name, _.description, pos("gene_name")/"node_pos").copy()
    #create network based on node ids
    network = Network(node_ids.nodeid())

    print "Adding node attributes..."
    network.addNodeAttributes("description", node_ids.description())
    network.addNodeAttributes("gene_name", node_ids.gene_name())

    addprotein_dna_edges(network, gn_to_orf)
    add_protprot_edges(network, gn_to_orf)
   
    print "Adding measurement edges..."
    #add gene node to measurement node edges
    cmean = [numpy.mean(row[~numpy.isnan(row)]) for row in cgenes.intensity()]
    cstd = [numpy.std(row[~numpy.isnan(row)]) ** 2 for row in cgenes.ratio()]
    network.addEdges("measurement", cgenes.orf_code(), cgenes.orf_code(), numpy.array([cmean,  cstd]).T, ["intensity", "std"], [0,0])

    #TARGET DATA

    kdata = (wrapper_tsv.read_tsv('/home/marc/Databases/yeast_knockouts/p_knockouts.txt')/('knockout_name', 'target_name','p_value'))
    kdata = kdata.match(node_ids, "target_name", "nodeid").get(_.nodeid/"target_orf", _.knockout_name, _.p_value, _.node_pos/"gene_pos")
    kdata = kdata.match(node_ids, "knockout_name", "nodeid").get(_.nodeid/"knockout_orf", _.target_orf, _.p_value, _.gene_pos).copy()
    kdata = kdata.group_by(_.target_orf, _.knockout_orf, keep={(0,):("gene_pos",), (0,1):("p_value",)}).copy()

    target_dictn = {}
    for node,pvals in zip(kdata.knockout_orf(), kdata.p_value().T):
        target_dictn[node] = 1 - pvals

    #determine position of gene names in nodeids list(network), by matching them with positioned node list, 
    #sorting result on gene pos (thus using order in pvals), and returning node pos list
    target_dictn["__sel__"] = kdata.gene_pos()


    #filter experiment list on knockouts that are not part of the titration series
    print "Loading experiment data and use it to translate knockout names..."
    experiment_list = experiment_list[_.repeated.map(lambda x: not "titration" in x, otype="bool")]

    #Create a position for each knockout (to use as index after joining). Match knockouts in hughes data with hughes experiment list data to get orf codes. 
    #knockouts = hughes_data.knockouts.row_dict()[experiment_list.experiment_name].hstack(experiment_list.orf_code)
    knockouts = hughes_data.knockouts.get(pos()/"pos", _).match(experiment_list, "knockouts", "experiment_name")
    
    #Translate orf codes to upper case, keep only those rows for which the orf_code is unique
    #knockouts = knockouts.get(_.orf_code.map(str.upper, otype="bytes[]"), "~").over(group=_.orf_code, do=_[_.experiment_name.count() == 1])
    knockouts = knockouts.get(_.orf_code.map(str.upper, otype="bytes[]"), "~").group_by(_.orf_code)[_.experiment_name.count() == 1].flat()

    #match knockouts with genelist orf codes (to filter out non-existing), keep position and orf_code
    #knockouts = knockouts[_.orf_code.within(gn.orf_code)]
    print "Matching knockouts to orf names..."
    knockouts = knockouts.match(gn, "orf_code", "orf_code").get(_.pos, _.orf_code, _.experiment_name).copy()

    #get gene names from each measured transcript in hughes data, note position of gene name
    #then match with genenames, keep gene official name (orf code) and position

    print "Loading knockout pvalues and creating target data..."
    genes = hughes_data.match(node_ids, "gene_official", "nodeid")[ismissing(_.pval).sum() == 0].copy()
    pvals = genes.pval()
    
    #replace pvals == 1 with NaN
    sel_missing = numpy.equal(pvals,1)
    pvals[sel_missing] = numpy.nan

    target_dict = {}
    for node,knockout_pos in zip(knockouts.orf_code(), knockouts.pos()):
        target_dict[node] = 1 - pvals[:, knockout_pos]

    #determine position of gene names in nodeids list(network), by matching them with positioned node list, 
    #sorting result on gene pos (thus using order in pvals), and returning node pos list
    target_dict["__sel__"] = genes.node_pos()
    return (network, target_dict, target_dictn)


