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
          Head
        </Typography>
        <Typography component="p">
          Returns the latest (head) version of a key's particular branch
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'HEAD -k <key> -b <branch>'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._b}
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
ustore> head -k myfirstKey -b named-merge-with-edit
[SUCCESS: HEAD] Version: GV4MY4NVTIXI662AQKDNNUTDGHBTSCIX
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
