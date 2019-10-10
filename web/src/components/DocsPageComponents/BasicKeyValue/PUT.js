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
          Put
        </Typography>
        <Typography component="p">
          Put the value to a key
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'PUT -k <key> [-x <value> | <file>] {-p <type>} {-b <branch> | -u <refer_version>}'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._x}
        {GeneralOptions._file}
        {GeneralOptions._p}
        {GeneralOptions._b}
        {GeneralOptions._u}
        <Typography component="p">
          Utility Options:
        </Typography>
        {UtilityOptions._none}
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
// Put simple string to master
ustore> put -k myfirstKey -b master -x "this is the first key i put into the rafiki"
[SUCCESS: PUT] Type: Blob, Version: QOPJBWT4ITXVOSFWY4RNYPIXIPRVHCAZ

// Put a file to master
ustore> put -k File1 -b master ../mock-data/sample.csv
[SUCCESS: PUT] Type: Blob, Version: K6BVFFAYM3Z4JGCKYSAABPO4DBVLO4T3

ustore> put -k noBranchSpecified -x "if you do not specifiy the branch or version, it will be identified by the version"
[SUCCESS: PUT] Type: Blob, Version: 25J43M533ZI4GGWUUEAF6P3XZOS7KFJR
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
