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
          Merge
        </Typography>
        <Typography component="p">
          Merge two versions of a key into one by creating a new value. "merge -b target_branch -c refer_branch" will create new node after target_branch
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'MERGE -k <key> -x <value> [-b <target_branch> -c <refer_branch> | -b <target_branch> -u <refer_version> | -u <refer_version> -v <refer_version_2>]'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._x}
        {GeneralOptions._b}
        {GeneralOptions._c}
        {GeneralOptions._u}
        {GeneralOptions._v}
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
ustore> merge -k myfirstKey -x "this is not a COMMIT message! this is the actual value! merge named-merge-no-edit with master" -b master -c named-merge-no-edit
[SUCCESS: MERGE] Version: F5RZHEHBVWO5IFM5NUAPY75LEPH7INMR

ustore> merge -k myfirstKey -x "merge -x will write new value. (merge named-merge-with-edit with master)" -b master -c named-merge-with-edit
[SUCCESS: MERGE] Version: IDAJSCWZSU2RW63WMSH3T4FCEC4ZJL3K
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
