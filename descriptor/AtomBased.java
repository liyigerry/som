import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.atomtype.SybylAtomTypeMatcher;
import org.openscience.cdk.graph.ShortestPaths;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IAtomType;
import org.openscience.cdk.io.SDFWriter;
import org.openscience.cdk.io.iterator.IteratingSDFReader;
import org.openscience.cdk.qsar.AbstractAtomicDescriptor;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.ArrayList;

public class AtomBased {

    public static HashMap<String, AbstractAtomicDescriptor> atomicDescriptorMap = new HashMap<String, AbstractAtomicDescriptor>();

    static {
        atomicDescriptorMap.put("EffectiveAtomPolarizability", new org.openscience.cdk.qsar.descriptors.atomic.EffectiveAtomPolarizabilityDescriptor());
        atomicDescriptorMap.put("StabilizationPlusCharge", new org.openscience.cdk.qsar.descriptors.atomic.StabilizationPlusChargeDescriptor());
        atomicDescriptorMap.put("SigmaElectronegativity", new org.openscience.cdk.qsar.descriptors.atomic.SigmaElectronegativityDescriptor());
        atomicDescriptorMap.put("PiElectronegativity", new org.openscience.cdk.qsar.descriptors.atomic.PiElectronegativityDescriptor());
        atomicDescriptorMap.put("PartialSigmaCharge", new org.openscience.cdk.qsar.descriptors.atomic.PartialSigmaChargeDescriptor());
        atomicDescriptorMap.put("PartialTChargeMMFF94", new org.openscience.cdk.qsar.descriptors.atomic.PartialTChargeMMFF94Descriptor());
        atomicDescriptorMap.put("AtomDegree", new org.openscience.cdk.qsar.descriptors.atomic.AtomDegreeDescriptor());
        atomicDescriptorMap.put("AtomValance", new org.openscience.cdk.qsar.descriptors.atomic.AtomValenceDescriptor());
        atomicDescriptorMap.put("AtomHybridizationVSEPR", new org.openscience.cdk.qsar.descriptors.atomic.AtomHybridizationVSEPRDescriptor());
        atomicDescriptorMap.put("AtomHybridization", new org.openscience.cdk.qsar.descriptors.atomic.AtomHybridizationDescriptor());
        // new
        try {
            atomicDescriptorMap.put("CovalentRadius", new org.openscience.cdk.qsar.descriptors.atomic.CovalentRadiusDescriptor());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        // InductiveAtomicHardnessDescriptor
        try {
            atomicDescriptorMap.put("indAtomHardness", new org.openscience.cdk.qsar.descriptors.atomic.InductiveAtomicHardnessDescriptor());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        // InductiveAtomicSoftnessDescriptor
        try {
            atomicDescriptorMap.put("indAtomSoftness", new org.openscience.cdk.qsar.descriptors.atomic.InductiveAtomicSoftnessDescriptor());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        // PartialPiChargeDescriptor
        atomicDescriptorMap.put("pepe", new org.openscience.cdk.qsar.descriptors.atomic.PartialPiChargeDescriptor());
        // PartialTChargePEOEDescriptor
        atomicDescriptorMap.put("pepeT", new org.openscience.cdk.qsar.descriptors.atomic.PartialTChargePEOEDescriptor());
        // ProtonTotalPartialChargeDescriptor
        atomicDescriptorMap.put("protonTotalPartialCharge", new org.openscience.cdk.qsar.descriptors.atomic.ProtonTotalPartialChargeDescriptor());
        // VdWRadiusDescriptor
        try {
            atomicDescriptorMap.put("VdWRadius", new org.openscience.cdk.qsar.descriptors.atomic.VdWRadiusDescriptor());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {

        FileInputStream fis = null;
        IteratingSDFReader isr = null;
        SybylAtomTypeMatcher typeMatcher = SybylAtomTypeMatcher.getInstance(DefaultChemObjectBuilder.getInstance());

        // change here
        File file = new File("D:\\p450data\\split_dataset\\merged");
        File[] files = file.listFiles();
        int cnt = 1;
        try {
            for (File file1 : files) {
                fis = new FileInputStream(new File(file1.getPath()));
                isr = new IteratingSDFReader(fis, DefaultChemObjectBuilder.getInstance());
                while (isr.hasNext()) {
                    ArrayList<AtomInfo> atomInfoList = new ArrayList<AtomInfo>();
                    IAtomContainer mol = (IAtomContainer) isr.next();

                    System.out.println("*** " + (cnt++) + " *** " + mol.getProperty("ID"));
                    FileWriter sdfFr = new FileWriter("./merged/" + mol.getProperty("ID") + ".sdf");
                    SDFWriter sdfWriter = new SDFWriter(sdfFr);
                    //add H for calculate
                    AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
                    CDKHydrogenAdder.getInstance(mol.getBuilder()).addImplicitHydrogens(mol);
                    AtomContainerManipulator.convertImplicitToExplicitHydrogens(mol);
                    // find atom type
                    IAtomType[] atomTypes = typeMatcher.findMatchingAtomTypes(mol);

                    // for saving path
                    int[][] shortestPaths = new int[mol.getAtomCount()][mol.getAtomCount()];

                    // init longest dist
                    int longestMaxTopDistInMolecule = Integer.MIN_VALUE;

                    // new array save atomInfo
                    AtomInfo[] atomInfoArrayByMolecule = new AtomInfo[mol.getAtomCount()];

                    // save feature into AtomInfo
                    for (IAtom atom : mol.atoms()) {
                        // save AtomInfo
                        AtomInfo atomInfo = new AtomInfo();
                        atomInfoArrayByMolecule[atom.getIndex()] = atomInfo;
                        atomInfo.setTitle(mol.getProperty("ID"));
                        atomInfo.setIndex(atom.getIndex());
                        atomInfo.setSymbol(atom.getSymbol());
                        if (atomTypes[atom.getIndex()] != null) {
                            atomInfo.setAtomType(atomTypes[atom.getIndex()].getAtomTypeName());
                        }

                        // compute descriptor
                        for (String descriptorName : atomicDescriptorMap.keySet()) {
                            AbstractAtomicDescriptor aad = atomicDescriptorMap.get(descriptorName);
                            DescriptorValue dv = aad.calculate(atom, mol);
                            String aadValue = dv.getValue().toString();
                            atomInfo.setDescriptor(descriptorName, aadValue);
                        }

                        // compute path descriptor
                        ShortestPaths sp = new ShortestPaths(mol, atom);
                        atomInfo.setHighestMaxTopDistInMatrixRow(Integer.MIN_VALUE);
                        for (IAtom atom_target : mol.atoms()) {
                            shortestPaths[atom.getIndex()][atom_target.getIndex()] = sp.distanceTo(atom_target);
                            // ignore H
                            if (!atom.getSymbol().equals("H") && !atom_target.getSymbol().equals("H")) {
                                if (sp.distanceTo(atom_target) > atomInfo.getHighestMaxTopDistInMatrixRow()) {
                                    atomInfo.setHighestMaxTopDistInMatrixRow(sp.distanceTo(atom_target));
                                }
                                if (sp.distanceTo(atom_target) > longestMaxTopDistInMolecule) {
                                    longestMaxTopDistInMolecule = sp.distanceTo(atom_target);
                                }
                            }
                        }
                    }

                    // set longest path value
                    for (int i = 0; i < shortestPaths.length; i++) {
                        atomInfoArrayByMolecule[i].setLongestMaxTopDistInMolecule(longestMaxTopDistInMolecule);
                    }

                    // save to atomInfoList
                    for (int i = 0; i < atomInfoArrayByMolecule.length; i++) {
                        if (!atomInfoArrayByMolecule[i].getSymbol().equals("H")) {
                            atomInfoList.add(atomInfoArrayByMolecule[i]);
                        }
                    }

                    // build feature string, save as mol attribute
                    String longestMaxTopInMoleculeStr = "";
                    String highestMaxTopInMoleculeStr = "";
                    String diffSPAN3Str = "";
                    String relSPAN4Str = "";
                    String atomTypeStr = "";
                    for (int i = 0; i < atomInfoList.size(); i++) {
                        if (i != 0) {
                            longestMaxTopInMoleculeStr += " ";
                            highestMaxTopInMoleculeStr += " ";
                            diffSPAN3Str += " ";
                            relSPAN4Str += " ";
                            atomTypeStr += " ";
                        }
                        atomTypeStr += atomInfoList.get(i).getAtomType();
                        longestMaxTopInMoleculeStr += atomInfoList.get(i).getLongestMaxTopDistInMolecule();
                        highestMaxTopInMoleculeStr += atomInfoList.get(i).getHighestMaxTopDistInMatrixRow();
                        diffSPAN3Str += atomInfoList.get(i).getLongestMaxTopDistInMolecule() - atomInfoList.get(i).getHighestMaxTopDistInMatrixRow();
                        relSPAN4Str += String.valueOf((double) atomInfoList.get(i).getHighestMaxTopDistInMatrixRow() / (double) atomInfoList.get(i).getLongestMaxTopDistInMolecule());
                    }

                    mol.setProperty("atomType", atomTypeStr);
                    mol.setProperty("longestMaxTopInMolecule", longestMaxTopInMoleculeStr);
                    mol.setProperty("highestMaxTopInMolecule", highestMaxTopInMoleculeStr);
                    mol.setProperty("diffSPAN3", diffSPAN3Str);
                    mol.setProperty("relSPAN4", relSPAN4Str);

                    for (String descriptorName : atomicDescriptorMap.keySet()) {
                        String data = "";

                        for (int i = 0; i < atomInfoList.size(); i++) {
                            if (i != 0) {
                                data += " ";
                            }
                            data += (String) atomInfoList.get(i).getDescriptor(descriptorName);
                        }
                        mol.setProperty(descriptorName, data);
                    }
                    sdfWriter.write(mol);
                    sdfWriter.close();
                    sdfFr.close();
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }

    }
}
