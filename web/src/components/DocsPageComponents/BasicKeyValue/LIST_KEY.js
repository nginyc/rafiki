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
          List Key
        </Typography>
        <Typography component="p">
          List all the current keys, including the meta-data from datasets and tables, like $xyz
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'LIST_KEY{_ALL}'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._none}
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
ustore> list_key
[SUCCESS: LIST_KEY] Keys: ["File1", "myfirstKey", "noBranchSpecified"]

// -1 flag list in vertial list view
ustore> list_key_all -1
File1
myfirstKey
noBranchSpecified
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
