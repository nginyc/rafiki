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
          Latest
        </Typography>
        <Typography component="p">
          Returns in a list form all the latest (head) versions of a key, ie the leaf-nodes. If the list of latest versions is too long, "LATEST" will only display the first 20 versions. To see all the versions in the list, use "LATEST_ALL".
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'LATEST{_ALL} -k <key>'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._b}
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
ustore> latest -k myfirstKey
[SUCCESS: LATEST] Versions: [IDAJSCWZSU2RW63WMSH3T4FCEC4ZJL3K]

ustore> latest -k noBranchSpecified
[SUCCESS: LATEST] Versions: [RNCK2CFCL3KGVUZ6RVKJHL3B7QUTTIC7]

ustore> latest -k File1
[SUCCESS: LATEST] Versions: [K6BVFFAYM3Z4JGCKYSAABPO4DBVLO4T3]
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
