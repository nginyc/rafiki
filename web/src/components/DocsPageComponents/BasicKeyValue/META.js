import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

import { GeneralOptions, UtilityOptions } from "../Options"

const styles = {
  card: {
    maxWidth: "100%",
  },
};

function DocsCard(props) {
  const { classes } = props;
  return (
    <Card className={classes.card}>
      <CardContent>
        <Typography gutterBottom variant="h3" component="h1">
          Meta
        </Typography>
        <Typography component="p">
          Meta gives the Type, Value, Version, and parents
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'META -k <key> [-b <branch> | -v <version>]'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._b}
        {GeneralOptions._v}
        <Typography component="p">
          Utility Options:
        </Typography>
        {UtilityOptions._1}
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> meta -k noBranchSpecified -v RNCK2CFCL3KGVUZ6RVKJHL3B7QUTTIC7
[SUCCESS: META] Type: Blob, Value: "if you do not specifiy the branch or version, its parent will be Null", Version: RNCK2CFCL3KGVUZ6RVKJHL3B7QUTTIC7, Parents: [<null>]

ustore> meta -k File1 -b master
[SUCCESS: META] Type: Blob, Value: "KEY,AGE,GENDER,GPA,SCHOOL
a01,20,Male,4.9,NUS
a02,22,Male,2.9,NTU
a03,28,Female,4.2,NUS
a04,39,Female,3.9,SMU
a05,16,Male,5.0,NUS", Version: K6BVFFAYM3Z4JGCKYSAABPO4DBVLO4T3, Parents: [<null>]

// -1 in list view
ustore> meta -k myfirstKey -b master -1
Type   : String
Value  : "merge -x will write new value. (merge named-merge-with-edit with master)"
Version: IDAJSCWZSU2RW63WMSH3T4FCEC4ZJL3K
Parents: [F5RZHEHBVWO5IFM5NUAPY75LEPH7INMR, GV4MY4NVTIXI662AQKDNNUTDGHBTSCIX]
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
